cmake_minimum_required(VERSION 3.1)

#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 -gencode arch=compute_52,code=sm_52)
#SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
project(selika_server)

set(SELIKA_VERSION_MAJOR 0)
set(SELIKA_VERSION_MINOR 1)
set(SELIKA_VERSION_PATCH 0)

add_subdirectory(src)
