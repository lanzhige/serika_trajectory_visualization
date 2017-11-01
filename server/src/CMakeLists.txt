set(CMAKE_CXX_STANDARD 11)

configure_file (
  "./selika.h.in"
  "${PROJECT_BINARY_DIR}/selika.h"
)

include_directories("${PROJECT_SOURCE_DIR}/src" "${PROJECT_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/include")
link_directories("${PROJECT_SOURCE_DIR}/lib")
set(CMAKE_PROGRAM_PATH "${PROJECT_SOURCE_DIR}/bin")
set(CMAKE_LIBRARY_PATH "${PROJECT_SOURCE_DIR}/lib")
add_definitions(-D_WIN32_WINNT=0x0501)


SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

option(BUILD_SERVER "Build server" ON)

add_subdirectory(server)
