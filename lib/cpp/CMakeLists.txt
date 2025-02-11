set(CMAKE_CXX_STANDARD 20)
set(LIB_NAME poker_rnad_cpp)
add_library(${LIB_NAME})

target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_sources(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ActorWorker.cpp)

target_include_directories(${LIB_NAME} PUBLIC ${poker-rnad_SOURCE_DIR}/contrib/open_spiel)
target_include_directories(${LIB_NAME} PUBLIC ${poker-rnad_SOURCE_DIR}/contrib/open_spiel/open_spiel/abseil-cpp)

# [TODO] fix this hack
if (APPLE)
    list(APPEND CMAKE_PREFIX_PATH "/Users/penguin-diver/miniconda3/lib/python3.12/site-packages/torch/share/cmake/Torch")
elseif(UNIX)
    add_library(open_spiel SHARED IMPORTED)
    set(OPEN_SPIEL_PATH "/opt/conda/lib/python3.11/site-packages/pyspiel.so")
    set_target_properties(open_spiel PROPERTIES IMPORTED_LOCATION ${OPEN_SPIEL_PATH})
    target_link_libraries(${LIB_NAME} PUBLIC open_spiel)
endif()

find_package(Torch REQUIRED)
target_include_directories(${LIB_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(${LIB_NAME} PUBLIC ${TORCH_LIBRARIES} thread-pool)
