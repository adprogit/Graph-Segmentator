cmake_minimum_required(VERSION 3.20)
project(pyplus CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Pas de contraction FMA : les flottants doivent etre arrondis exactement
# comme en Python/NumPy pour que la validation croisee soit bit-a-bit
# (sinon a*b + c*d fusionne en fma() et les seuils de decision divergent)
add_compile_options(-ffp-contract=off)

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
    target_sources(pyplus_core PRIVATE
        src/features.cc
        src/segmentation.cc
        src/pipeline.cc
        src/export_table.cc
        src/split_merged_tips.cc
    )
    # seulement les modules utilises : ${OpenCV_LIBS} lierait ~60 dylibs
    # et multiplie par ~6 le temps de demarrage du binaire (135ms -> 21ms)
    target_link_libraries(pyplus_core PUBLIC
        opencv_core opencv_imgproc opencv_imgcodecs)
    target_include_directories(pyplus_core PUBLIC ${OpenCV_INCLUDE_DIRS})
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
# fixtures regenerables : uv run --project python cpp/tests/gen_cross_fixture.py
add_test(NAME cross_letters COMMAND test_cross_python
    ${CMAKE_CURRENT_SOURCE_DIR}/../data/knn_letters.bin
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/cross_predictions_letters.txt
)
add_test(NAME cross_digits COMMAND test_cross_python
    ${CMAKE_CURRENT_SOURCE_DIR}/../data/knn_digits.bin
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/cross_predictions_digits.txt
)

if(OpenCV_FOUND)
    foreach(name test_features test_features_cross test_segmentation
                 test_seg_cross test_export_table test_table_cross)
        add_executable(${name} tests/${name}.cc)
        target_compile_options(${name} PRIVATE -Wall -Wextra)
        target_link_libraries(${name} PRIVATE pyplus_core)
    endforeach()
    add_test(NAME features COMMAND test_features)
    # fixture : uv run --project python cpp/tests/gen_features_fixture.py
    add_test(NAME features_cross COMMAND test_features_cross
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/features_expected.txt
    )
    add_test(NAME segmentation COMMAND test_segmentation)
    add_test(NAME export_table COMMAND test_export_table)
    # fixtures : uv run --project python cpp/tests/gen_seg_fixture.py
    add_test(NAME seg_cross COMMAND test_seg_cross
        ${CMAKE_CURRENT_SOURCE_DIR}/../data/knn_letters.bin
        ${CMAKE_CURRENT_SOURCE_DIR}/../data/knn_digits.bin
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/seg
    )
    add_test(NAME table_cross COMMAND test_table_cross
        ${CMAKE_CURRENT_SOURCE_DIR}/../data/knn_letters.bin
        ${CMAKE_CURRENT_SOURCE_DIR}/../data/knn_digits.bin
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/seg
    )
    add_test(NAME cli_batch COMMAND pyplus
        --batch ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/seg
        --out ${CMAKE_CURRENT_BINARY_DIR}/batch_out
        --letters ${CMAKE_CURRENT_SOURCE_DIR}/../data/knn_letters.bin
        --digits ${CMAKE_CURRENT_SOURCE_DIR}/../data/knn_digits.bin
    )
endif()
