cmake_minimum_required(VERSION 3.10)
project(geminigraph)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 设置编译器为 mpicxx
# set(CMAKE_CXX_COMPILER mpicxx)

# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE Release)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 设置默认值，确保 OS 始终被定义
set(OS DEFAULT)

if(CMAKE_C_COMPILER MATCHES "musl-gcc")
    set(OS CHCORE)
endif()

# 设置编译选项（不在这里加 PIE，由每个 target 单独控制）
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -march=native -O3")
# 如果需要启用宏定义，取消下面行的注释
# add_definitions(-DPRINT_DEBUG_MESSAGES)

# 包含头文件目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# 查找所有头文件
file(GLOB HEADERS "*.hpp")

# 定义目标
set(TARGETS bc bfs cc pagerank sssp)

# 添加每个工具的可执行文件
foreach(target ${TARGETS})
    add_executable(${target} toolkits/${target}.cpp toolkits/parallel.cpp ${HEADERS})
    if (${OS} STREQUAL "CHCORE")
        if (CHCORE_USER_STATIC_APPS)
            target_link_options(${target} PRIVATE -static)
            target_link_libraries(${target} PRIVATE rpmalloc-static libc-static libstdc++-static)
        else()
            target_link_libraries(${target} PRIVATE rpmalloc-shared)
            target_compile_options(${target} PRIVATE -fPIE -fPIC)
            target_link_options(${target} PRIVATE -pie)
        endif()
        target_compile_definitions(${target} PRIVATE DSM_ENABLED OS_CHCORE)
    endif()
endforeach()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

if(DEFINED OS AND ${OS} STREQUAL "CHCORE")
    add_definitions(-DOS_CHCORE)
    add_custom_target(
        gemini-copy-data
        COMMAND rsync -a ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/pagerank ${build_ramdisk_dir}/
        COMMAND rsync -a ${CMAKE_CURRENT_SOURCE_DIR}/cnr-2000.binedgelist ${build_ramdisk_dir}/
        DEPENDS ${TARGETS}
    )
    add_dependencies(ramdisk gemini-copy-data)
endif()
