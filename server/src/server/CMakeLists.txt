set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOMOC ON)

# source
###############################################################################
file(GLOB hdrs "*.h" "*.hpp" "../cuda/*.h")
file(GLOB srcs "*.cpp" "../cuda/*.cu")

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

find_package(Boost COMPONENTS system thread program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package(CUDA)

if(CUDA_FOUND)
  # Look for Thrust in CUDA directories.
  find_path(CUDATHRUST_INCLUDE
    thrust/version.h
    PATHS ${CUDA_INCLUDE_DIRS} ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_SDK_ROOT_DIR}
    NO_DEFAULT_PATH
    )
  # Look for Thrust in default search paths.
  find_path(CUDATHRUST_INCLUDE thrust/version.h)
  mark_as_advanced(CUDATHRUST_INCLUDE)
  if(CUDATHRUST_INCLUDE)
    # Thrust was found.
    set(CUDATHRUST_FOUND TRUE)
    set (CUDATHRUST_INCLUDE_DIRS ${CUDATHRUST_INCLUDE})
  else(CUDATHRUST_INCLUDE)
    # Thrust was not found.
    set(CUDATHRUST_FOUND FALSE)
    if(CUDATHRUST_FIND_REQUIRED)
      message(FATAL_ERROR "CUDA Thrust not found!")
    else(CUDATHRUST_FIND_REQUIRED)
      if (NOT CUDATHRUST_FIND_QUIETLY)
        message(STATUS "CUDA Thrust not found")
      endif(NOT CUDATHRUST_FIND_QUIETLY)
    endif(CUDATHRUST_FIND_REQUIRED)
  endif(CUDATHRUST_INCLUDE)
else(CUDA_FOUND)
  if(NOT CUDATHRUST_FIND_QUIETLY)
    message(STATUS "CUDA must be found before CMake looks for Thrust!")
  endif(NOT CUDATHRUST_FIND_QUIETLY)
  set(CUDATHRUST_FOUND FALSE)
endif(CUDA_FOUND)


find_package(OpenSSL)
find_package(ZLIB)
#if (EXIST "/user/bin/gcc-4.7") set(CUDA_HOST_COMPILER /usr/bin/gcc-4.7)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52)

cuda_add_executable(server ${hdrs} ${srcs})
target_link_libraries(server ${Boost_LIBRARIES})
