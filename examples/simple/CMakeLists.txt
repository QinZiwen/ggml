#
# simple-ctx

set(TEST_TARGET simple-ctx)
add_executable(${TEST_TARGET} simple-ctx.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)

#
# simple-backend

set(TEST_TARGET simple-backend)
add_executable(${TEST_TARGET} simple-backend.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)

set(TEST_TARGET simple-demo)
add_executable(${TEST_TARGET} simple-demo.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)

find_library(ACCELERATE_FRAMEWORK Accelerate)
if (NOT ACCELERATE_FRAMEWORK)
    message(FATAL_ERROR "未找到 Mac Accelerate 框架，无法启用硬件加速！")
endif()
set(TEST_TARGET simple-benchmark)
add_executable(${TEST_TARGET} benchmark.c)
target_link_libraries(${TEST_TARGET} PRIVATE ggml ${ACCELERATE_FRAMEWORK})
target_compile_definitions(${TEST_TARGET} PRIVATE
    GGML_USE_ACCELERATE
    ACCELERATE_NEW_LAPACK
    ACCELERATE_LAPACK_ILP64
)

if (GGML_CUDA)
    add_compile_definitions(GGML_USE_CUDA)
endif()

if (GGML_METAL)
    add_compile_definitions(GGML_USE_METAL)
endif()
