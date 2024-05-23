cmake_minimum_required(VERSION 3.11)

project(Negate CXX)
set(CMAKE_CXX_STANDARD 17)

set(RECOGNIZER_FILES
    main.cpp
    neural_matrix.h neural_matrix.cpp
    training_database.h training_database.cpp
    profiler.h)

add_subdirectory(ImgLib ImgLibBuildDir)

add_executable(recognizer ${RECOGNIZER_FILES})
target_include_directories(recognizer PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/ImgLib")
target_link_libraries(recognizer ImgLib)