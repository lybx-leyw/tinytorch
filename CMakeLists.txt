cmake_minimum_required(VERSION 3.10)
project(MyDarknet C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W3")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Zi /Od /DDEBUG")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /O2 /DNDEBUG")
    endif()
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0 -DDEBUG")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -DNDEBUG")
    endif()
endif()

# 核心源文件
file(GLOB SOURCES "src/*.c")
file(GLOB NN_SOURCES "src/nn/*.c")
file(GLOB QUICK_SOURCES "src/quike_start/*.c")

# 添加核心源文件
list(APPEND SOURCES ${NN_SOURCES})
list(APPEND SOURCES ${QUICK_SOURCES})

# 示例文件 - 默认只编译 example_2 中的 main.c
file(GLOB EXAMPLE_2_SOURCES "src/examples/example_2/*.c")
list(APPEND SOURCES ${EXAMPLE_2_SOURCES})

# 头文件
file(GLOB HEADERS "include/*.h")

# 创建可执行文件
add_executable(my_program ${SOURCES})
target_include_directories(my_program PRIVATE include)
