This is just a basic cpp server using uWebsockets that connects to the raspberry pi camera module using libcamera. It's the complex version of just using python with picamera2. There's no AI or anything fancy, just a websocket connection that streams down video frames from the raspberry pi camera. Tested on a raspberry pi 5 only.

## Build

```
cmake -B build -S src -G Ninja
cmake --build build
```

## Requirements

upgrade cmake

```
wget https://github.com/Kitware/CMake/releases/download/v3.30.1/cmake-3.30.1.tar.gz \
    && tar -zxvf cmake-3.30.1.tar.gz \ 
    && cd cmake-3.30.1 \
    && ./bootstrap && make && make install 
```

install most up to date clang from source

```
RUN apt-get update && apt-get install -y swig libedit-dev libncurses5-dev libxml2-dev python3-dev liblua5.3-dev build-essential \
    && git clone https://github.com/llvm/llvm-project.git \
    && cd llvm-project \
    && mkdir build \
    && cd build \
    && export CXXFLAGS="$CXXFLAGS -Wno-error" \
    && cmake -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;lldb;lld" \
        -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;libunwind" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_TARGETS_TO_BUILD="AArch64" \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -G "Unix Makefiles" \
        ../llvm \
    && make -j$(nproc) \
    && make install
```

install uWebSockets

```
git clone https://github.com/uNetworking/uWebSockets.git \
 && cd uWebSockets 
``

Update the install instruction and the flags in the `GNUMakefile`

CXX = clang++
CXXFLAGS = -std=c++23 -stdlib=libc++ -fmodules -fbuiltin-module-map -fimplicit-modules -fimplicit-module-maps -Wall -Wextra -pedantic

install:
	mkdir -p "$(DESTDIR)$(prefix)/include/uWebSockets"
	cp -r src/* "$(DESTDIR)$(prefix)/include/uWebSockets"
	cp uSockets/src/libusockets.h "$(DESTDIR)$(prefix)/include"
	cp uSockets/uSockets.a "$(DESTDIR)$(prefix)/lib"
```

```
WITH_LTO=0 SSL_DISABLED=1 make \
&& WITH_LTO=0 SSL_DISABLED=1 make install
```

## Client

The client needs to send a post request to `/start` or `/stop` to start or stop the stream. Any client connected to the websocket endoint `ws` will recieve image buffers to display the video feed.
