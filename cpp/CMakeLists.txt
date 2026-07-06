cmake_minimum_required(VERSION 3.20)
project(pyplus CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# OpenCV optionnel tant que le portage n'a pas commence (brew install opencv)
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "OpenCV ${OpenCV_VERSION} trouve")
else()
    message(WARNING "OpenCV introuvable : compilation du squelette seul")
endif()

add_executable(pyplus
    src/main.cc
)
target_include_directories(pyplus PRIVATE include)
target_compile_options(pyplus PRIVATE -Wall -Wextra)

if(OpenCV_FOUND)
    target_link_libraries(pyplus PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(pyplus PRIVATE HAVE_OPENCV)
endif()
