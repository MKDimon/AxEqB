cmake_minimum_required(VERSION 3.25.1 FATAL_ERROR)

project(tests VERSION 0.1.0 DESCRIPTION "tests description")

# Стандарты
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(${PROJECT_NAME} tests.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC AxEqB)
add_test(NAME tests COMMAND tests)
