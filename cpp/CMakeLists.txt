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
    message(WARNING "OpenCV introuvable : compilation sans le pipeline image")
endif()

add_library(pyplus_core
    src/model_io.cc
    src/classifier.cc
)
target_include_directories(pyplus_core PUBLIC include)
target_compile_options(pyplus_core PRIVATE -Wall -Wextra)

if(OpenCV_FOUND)
    target_link_libraries(pyplus_core PUBLIC ${OpenCV_LIBS})
    target_compile_definitions(pyplus_core PUBLIC HAVE_OPENCV)
endif()

add_executable(pyplus src/main.cc)
target_compile_options(pyplus PRIVATE -Wall -Wextra)
target_link_libraries(pyplus PRIVATE pyplus_core)

# ---------------------------------------------------------------------------
# Tests (ctest --test-dir build)
# ---------------------------------------------------------------------------
enable_testing()

foreach(name test_model_io test_classifier test_cross_python)
    add_executable(${name} tests/${name}.cc)
    target_compile_options(${name} PRIVATE -Wall -Wextra)
    target_link_libraries(${name} PRIVATE pyplus_core)
endforeach()

add_test(NAME model_io COMMAND test_model_io)
add_test(NAME classifier COMMAND test_classifier)
# fixture regenerable : uv run --project python cpp/tests/gen_cross_fixture.py
add_test(NAME cross_python COMMAND test_cross_python
    ${CMAKE_CURRENT_SOURCE_DIR}/../data/knn_model.bin
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/cross_predictions.txt
)
