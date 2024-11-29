cmake_minimum_required(VERSION 3.5)

project(AxEqB LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add_executable(AxEqB main.cpp
#     bicgstab.h bicgstab.cpp)

include(GNUInstallDirs)
include(CTest)

# Указываем директории
set(PROJECT_SOURCE_DIR "src")
set(PROJECT_INCLUDE_DIR "include")

# Заголовочные файлы
set(
    HEADERS
    ${PROJECT_INCLUDE_DIR}/bicgstab.h
)

# Исходные файлы
set(
    SOURCE
    ${PROJECT_SOURCE_DIR}/bicgstab.cpp
)

# Добавляем библиотеку
add_library(
    ${PROJECT_NAME}
    STATIC
    ${HEADERS}
    ${SOURCE}
)

# Добавляем директорию для возможности подключения
target_include_directories(${PROJECT_NAME} PUBLIC  include/)

# Добавляем поддержку библиотеки OpenMP
find_package(OpenMP REQUIRED)
if(NOT OpenMP_CXX_FOUND)
        message(SEND_ERROR "Failed to find OpenMP.")
        return()
else()
        include_directories(${OpenMP_INCLUDE_DIRS})
    message("OpenMP FOUND dir is ${OpenMP_INCLUDE_DIRS}.")
endif()
# Дополнительный флаг, необходим для функционирования библиотеки
set(CMAKE_CXX_FLAGS "-fopenmp")

# Указание подключаемых библиотек
target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)

# Подключение самих тестов
if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    add_subdirectory(tests)
endif ()


enable_testing()
