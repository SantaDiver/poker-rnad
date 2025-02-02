set(LIB_NAME poker_rnad_py)
pybind11_add_module(${LIB_NAME} pybind11.cpp)

target_compile_definitions(${LIB_NAME} PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

find_package(Torch REQUIRED)
target_include_directories(${LIB_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(${LIB_NAME} PRIVATE poker_rnad_cpp)
