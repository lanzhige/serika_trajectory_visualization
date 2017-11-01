#!/bin/sh
echo "installing the server in ./build/bin directory"

cd build
cmake ..
cmake --build .

cp ./bin/server ../

