cmake_minimum_required(VERSION 3.15)
project(ImageFlasherWGPU)
set(CMAKE_CXX_STANDARD 20)

message(STATUS "Using toolchain file: ${CMAKE_TOOLCHAIN_FILE}")

# Set the C++ compiler to em++
set(CMAKE_CXX_COMPILER em++)

include_directories(extern/glm EXCLUDE_FROM_ALL)

# Set the source files with new structure
set(SOURCES
        src/cpp/main.cpp
        src/cpp/stb_image.h
        src/cpp/stb_image_resize.h
)

# Add the executable
add_executable(index ${SOURCES})

# Set compiler and linker flags dont forget DISABLE_EXCEPTION_CATCHING=1
target_compile_options(index
        PRIVATE
        -Wall
        -Wformat
        #        -g4
        -O0
        -gsource-map
        -s
        USE_BOOST_HEADERS=1
        "SHELL:-s USE_PTHREADS=1"
        -pthread
)

# Set linker flags for Emscripten
target_link_options(index PRIVATE
#        "SHELL:-s USE_GLFW=3"
        "SHELL:-s USE_WEBGPU=1"
#        "SHELL:-s USE_WEBGL2=1"
        "SHELL:-s WASM=1"
        "SHELL:-s ALLOW_MEMORY_GROWTH=1"
        "SHELL:-s NO_EXIT_RUNTIME=0"

        "SHELL:-s ASSERTIONS=2"
        "SHELL:-s SAFE_HEAP=1"
        "SHELL:-s STACK_OVERFLOW_CHECK=2"

        "SHELL:-s DISABLE_EXCEPTION_CATCHING=0"

        # Add malloc and free here - let EMSCRIPTEN_KEEPALIVE handle function exports
        "SHELL:-s EXPORTED_RUNTIME_METHODS=['ccall','cwrap','stackAlloc','stackSave','stackRestore','HEAPU8']"
        "SHELL:-s EXPORTED_FUNCTIONS=['_malloc','_free','_main']"

#        "SHELL:-s USE_BOOST_HEADERS=1"
        -pthread
        "SHELL:-s PTHREAD_POOL_SIZE=10"
        "SHELL:-s FULL_ES3=1"
        # Generate source maps for debugging
        -gsource-map --source-map-base http://localhost:8000/
)

# Copy files to public directory (for Node.js structure)
configure_file(public/index.html public/index.html COPYONLY)
configure_file(public/app.js public/app.js COPYONLY)

# Conditionally set file system related flags
set(USE_FILE_SYSTEM 1) # Set this to 1 to enable file-system

target_link_options(index PRIVATE --no-heap-copy)

# Custom target for building and copying assets to public directory
add_custom_target(copy_assets ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/index.js ${CMAKE_CURRENT_SOURCE_DIR}/public/index.js
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/index.wasm ${CMAKE_CURRENT_SOURCE_DIR}/public/index.wasm
    DEPENDS index
    COMMENT "Copying WASM assets to public directory"
)

# Custom target for Node.js development
add_custom_target(serve
        COMMAND node server.js --ikeda
        DEPENDS copy_assets
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Starting Node.js server"
)
