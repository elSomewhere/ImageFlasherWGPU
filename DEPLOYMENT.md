# ImageFlasherWGPU Production Deployment Guide

## Overview

This guide covers deploying ImageFlasherWGPU in production with the Reddit crawler running server-side and multiple viewers accessing the stream simultaneously.

## Architecture

```
Internet → Load Balancer → Web Server → Image Pipeline → Redis Cache
                     ↓         ↓             ↓
                 Static Files  WebSocket   Processed
                              Streaming   Images
```

## Quick Start (Docker)

### 1. Basic Docker Deployment

```bash
# Clone the repository
git clone https://github.com/your-username/ImageFlasherWGPU.git
cd ImageFlasherWGPU

# Copy and configure environment
cp env.example .env
# Edit .env with your settings

# Build and start services
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs -f imageflasher
```

### 2. Production Deployment with Nginx

```bash
# Start with nginx proxy
docker-compose --profile production up -d

# Monitor logs
docker-compose logs -f
```

## Cloud Platform Deployments

### Option A: DigitalOcean App Platform

1. **Create App Platform project**
2. **Connect your GitHub repository**
3. **Configure build settings:**
   ```yaml
   name: imageflasher-wgpu
   services:
   - build_command: npm run build
     environment_slug: node-js
     github:
       branch: main
       deploy_on_push: true
     http_port: 8000
     instance_count: 1
     instance_size_slug: basic-xxs
     name: web
     run_command: node server.js --reddit
     source_dir: /
   ```

### Option B: AWS EC2 Deployment

```bash
# Launch EC2 instance (t3.small or larger)
# Install Docker and Docker Compose
sudo yum update -y
sudo yum install -y docker
sudo systemctl start docker
sudo usermod -a -G docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Deploy application
git clone https://github.com/your-username/ImageFlasherWGPU.git
cd ImageFlasherWGPU
cp env.example .env
# Configure .env file
docker-compose up -d
```

### Option C: Railway Deployment

1. **Connect GitHub repository to Railway**
2. **Add environment variables:**
   ```
   NODE_ENV=production
   REDDIT_SUBREDDIT=worldnews
   WEB_PORT=8000
   WEBSOCKET_PORT=5010
   ```
3. **Deploy automatically on push**

## Environment Configuration

### Critical Environment Variables

```bash
# Application
NODE_ENV=production
WEB_PORT=8000
WEBSOCKET_PORT=5010

# Reddit Configuration
REDDIT_SUBREDDIT=worldnews  # or cats, cyberpunk, art, etc.
REDDIT_MAX_PAGES=5          # Increase for longer streams
REDDIT_PAGE_DELAY=2.0       # Delay between page requests
REDDIT_IMAGE_DELAY=1.0      # Delay between image downloads

# Performance Tuning
MAX_QUEUE_SIZE=2000         # Larger queue for more viewers
SEND_DELAY=0.3              # Faster image streaming
MAX_CONCURRENT_DOWNLOADS=10 # More parallel downloads
```

## Scaling Configuration

### For High Traffic (100+ concurrent viewers)

1. **Horizontal Scaling:**
   ```yaml
   # docker-compose.override.yml
   services:
     imageflasher:
       deploy:
         replicas: 3
       environment:
         - REDIS_HOST=redis
   ```

2. **Load Balancer Configuration:**
   ```nginx
   upstream app_servers {
       server imageflasher_1:8000;
       server imageflasher_2:8000;
       server imageflasher_3:8000;
   }
   ```

3. **Redis Optimization:**
   ```bash
   # Increase Redis memory for image caching
   docker-compose exec redis redis-cli CONFIG SET maxmemory 1gb
   ```

## Performance Optimization

### 1. Image Pipeline Optimization

```python
# In scraper_3.py - production optimizations
SCRAPED_IMAGES = deque(maxlen=2000)  # Larger buffer
SEND_DELAY = 0.3  # Faster streaming
MAX_CONCURRENT_DOWNLOADS = 10  # Parallel downloads
```

### 2. WebSocket Optimization

```javascript
// Add to server.js
const WebSocket = require('ws');
const wss = new WebSocket.Server({ 
    port: 5010,
    perMessageDeflate: true,  // Compression
    maxPayload: 1024 * 1024   # 1MB max payload
});
```

### 3. CDN Integration

For global distribution, use CloudFlare or AWS CloudFront:

```nginx
# nginx.conf - add CDN headers
location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|wasm)$ {
    expires 1y;
    add_header Cache-Control "public, immutable";
    add_header CDN-Cache-Control "max-age=31536000";
}
```

## Monitoring and Maintenance

### 1. Health Checks

```bash
# Check application health
curl http://localhost:8000/health

# Check WebSocket connectivity
wscat -c ws://localhost:5010

# Monitor resource usage
docker stats
```

### 2. Log Management

```bash
# View logs
docker-compose logs -f imageflasher

# Log rotation (add to docker-compose.yml)
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

### 3. Backup Strategy

```bash
# Backup configuration
tar -czf backup-$(date +%Y%m%d).tar.gz \
    docker-compose.yml nginx.conf .env

# Database backup (if using persistent Redis)
docker-compose exec redis redis-cli BGSAVE
```

## Security Considerations

### 1. Reddit API Rate Limiting

```python
# Implement exponential backoff in scraper_3.py
import time
import random

def safe_request(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url)
            if response.status_code == 429:  # Rate limited
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
                continue
            return response
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt)
```

### 2. Resource Limits

```yaml
# docker-compose.yml - add resource limits
services:
  imageflasher:
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
```

### 3. HTTPS Configuration

```bash
# Get Let's Encrypt certificate
certbot --nginx -d your-domain.com

# Update nginx.conf with SSL settings
```

## Troubleshooting

### Common Issues

1. **WebGPU not working:**
   - Ensure CORS headers are properly set
   - Check browser WebGPU support
   - Verify HTTPS in production

2. **WebSocket connection failed:**
   - Check firewall settings for port 5010
   - Verify proxy configuration
   - Test direct WebSocket connection

3. **Reddit scraping blocked:**
   - Reduce scraping rate
   - Use different subreddit
   - Implement IP rotation

4. **High memory usage:**
   - Reduce MAX_QUEUE_SIZE
   - Implement image compression
   - Add Redis memory limits

### Debug Commands

```bash
# Test image pipeline
docker-compose exec imageflasher python3 src/python/scraper_3.py

# Test WebSocket
wscat -c ws://localhost:5010

# Check resource usage
docker-compose exec imageflasher top
docker-compose exec redis redis-cli INFO memory
```

## Cost Estimation

### Monthly Costs (USD)

| Platform | Small (10 users) | Medium (100 users) | Large (1000+ users) |
|----------|------------------|-------------------|---------------------|
| DigitalOcean | $25 | $50 | $200+ |
| AWS EC2 | $20 | $80 | $300+ |
| Railway | $15 | $40 | $150+ |

### Optimization Tips

1. **Use image compression** to reduce bandwidth costs
2. **Implement CDN** for static assets
3. **Cache processed images** in Redis
4. **Scale horizontally** instead of vertically when possible

## Production Checklist

- [ ] Environment variables configured
- [ ] SSL certificate installed
- [ ] Health checks implemented
- [ ] Log rotation configured
- [ ] Monitoring setup
- [ ] Backup strategy in place
- [ ] Resource limits set
- [ ] Rate limiting configured
- [ ] CORS headers properly set
- [ ] WebSocket proxy working
- [ ] Redis persistence configured
- [ ] Error handling implemented

## Support

For deployment issues:
1. Check the application logs
2. Verify all dependencies are installed
3. Test individual components separately
4. Review the troubleshooting section
5. Create an issue on GitHub with logs and configuration