cmake_minimum_required(VERSION 3.11)

project(Recognizer CXX)
set(CMAKE_CXX_STANDARD 17)

# Uncomment if maximum optimizations are needed
# (It boosts performance by 3.7% on the core i5 8600k)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

set(RECOGNIZER_FILES
    command_interpreter.h command_interpreter.cpp
    main.cpp
    profiler.h
    request_handler.h request_handler.cpp
    snn.h snn.cpp
    state_saver.h state_saver.cpp
    training_database.h training_database.cpp)

add_subdirectory(ImgLib ImgLibBuildDir)

add_executable(recognizer ${RECOGNIZER_FILES})
target_include_directories(recognizer PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/ImgLib")
target_link_libraries(recognizer ImgLib)