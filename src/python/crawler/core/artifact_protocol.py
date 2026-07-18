"""Versioned artifact framing shared by every producer and renderer client."""
from __future__ import annotations

import json
import struct
from dataclasses import asdict

from .types import Artifact


PROTOCOL_VERSION = 1
MAX_HEADER_BYTES = 64 * 1024


class ArtifactProtocolError(ValueError):
    pass


def artifact_header(artifact: Artifact) -> dict:
    header = {
        "protocol": PROTOCOL_VERSION,
        "session_id": artifact.session_id,
        "sequence": artifact.sequence,
        "kind": artifact.kind,
        "mime": artifact.mime,
        "source_url": artifact.source_url,
        "page_url": artifact.page_url,
        "acquired_at": artifact.acquired_at,
        "content_hash": artifact.content_hash,
        "width": artifact.width,
        "height": artifact.height,
        "duration": artifact.duration,
        "byte_size": len(artifact.payload),
        "score": round(float(artifact.score), 6),
        "novelty": round(float(artifact.novelty), 6),
        "producer": artifact.producer,
        "rights": asdict(artifact.rights),
    }
    if artifact.metadata:
        header["metadata"] = artifact.metadata
    return header


def encode_artifact(artifact: Artifact) -> bytes:
    encoded_header = json.dumps(
        artifact_header(artifact), separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    if len(encoded_header) > MAX_HEADER_BYTES:
        raise ArtifactProtocolError("Artifact header exceeds 64 KiB")
    return struct.pack("<I", len(encoded_header)) + encoded_header + artifact.payload


def decode_artifact_frame(frame: bytes) -> tuple[dict, bytes]:
    if len(frame) < 4:
        raise ArtifactProtocolError("Artifact frame is shorter than its length prefix")
    header_length = struct.unpack_from("<I", frame, 0)[0]
    if header_length <= 0 or header_length > MAX_HEADER_BYTES:
        raise ArtifactProtocolError("Invalid artifact header length")
    payload_offset = 4 + header_length
    if payload_offset > len(frame):
        raise ArtifactProtocolError("Artifact frame has a truncated header")
    try:
        header = json.loads(frame[4:payload_offset].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactProtocolError(f"Invalid artifact header: {error}") from error
    if header.get("protocol") != PROTOCOL_VERSION:
        raise ArtifactProtocolError(f"Unsupported protocol: {header.get('protocol')}")
    payload = frame[payload_offset:]
    if header.get("byte_size") != len(payload):
        raise ArtifactProtocolError("Artifact payload length does not match header")
    return header, payload
