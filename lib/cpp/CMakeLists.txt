set(LIB_NAME poker_rnad_cpp)
add_library(${LIB_NAME})

target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_sources(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Actor.cpp)

target_include_directories(${LIB_NAME} PUBLIC ${poker-rnad_SOURCE_DIR}/contrib/open_spiel)
target_include_directories(${LIB_NAME} PUBLIC ${poker-rnad_SOURCE_DIR}/contrib/open_spiel/open_spiel/abseil-cpp)
target_include_directories(${LIB_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})

list(APPEND CMAKE_PREFIX_PATH "${poker-rnad_SOURCE_DIR}/contrib/open_spiel/open_spiel/libtorch/libtorch")
find_package(Torch REQUIRED)

target_link_libraries(${LIB_NAME} PRIVATE
    # ${poker-rnad_SOURCE_DIR}/contrib/open_spiel/build/libopen_spiel.dylib
    ${TORCH_LIBRARIES}
)
