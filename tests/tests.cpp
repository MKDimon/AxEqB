#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include "bicgstab.h"

// Вывод матрицы в виде полной таблицы
void printMatrixCSR(int rows, int cols,
                    const std::vector<int>& row_ptr,
                    const std::vector<int>& col_indices,
                    const std::vector<double>& values) {
    // Для каждой строки
    for (int i = 0; i < rows; ++i) {
        int start = row_ptr[i];
        int end = row_ptr[i + 1];
        int col_index = 0;

        for (int j = 0; j < cols; ++j) {
            if (col_index < (end - start) && col_indices[start + col_index] == j) {
                // Если элемент есть в разреженной матрице
                std::cout << std::setw(6) << values[start + col_index] << " ";
                ++col_index;
            } else {
                // Если элемент равен 0
                std::cout << std::setw(6) << 0.0 << " ";
            }
        }
        std::cout << std::endl; // Переход на новую строку
    }
}

void test_with_params(const std::vector<int>& row_ptr, const std::vector<int>& col_indices,
                      const std::vector<double>& values, const std::vector<double>& b,
                      const int max_iter, const double tol,
                      const bool parallel, const bool print_answer = false) {

    std::vector<double> x(b.size(), 0); // Начальное приближение
    BiCGSTAB::BiCGSTAB worker(parallel);
    std::cout << (parallel ? "Parallel calc:" : "Not parallel calc") << std::endl;

    // Старт замера
    auto start = std::chrono::high_resolution_clock::now();
    if (worker.run(row_ptr, col_indices, values, b, x, max_iter, tol)) {
        // Конец замера
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "Time running: " << elapsed_seconds.count() << " sec" << std::endl;
        if (print_answer){
            std::cout << "Solution found: ";
            for (double val : x) {
                std::cout << val << " ";
            }
        }
        std::cout << std::endl;
    } else {
        std::cout << "Method did not converge in parallel try." << std::endl;
    }
}

// Функция для генерации разреженной ленточной матрицы в формате CSR
void generateSparseMatrixCSR(int n, int m,
                             std::vector<int>& row_ptr,
                             std::vector<int>& col_indices,
                             std::vector<double>& values) {
    row_ptr.resize(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        for (int j = std::max(0, i - m); j <= std::min(n - 1, i + m); ++j) {
            values.push_back(1.0); // Заполняем значением 1.0
            col_indices.push_back(j);
        }
        row_ptr[i + 1] = values.size();
    }
}

void test_with_generate_matrix(int n, int m, const int max_iter, const double tol, const bool parallel) {
    std::vector<int> row_ptr;           // Указатели строк
    std::vector<int> col_indices;       // Индексы столбцов
    std::vector<double> values;         // Значения
    std::vector<double> b(n, 1);        // Вектор b

    generateSparseMatrixCSR(n, m, row_ptr, col_indices, values);
    //printMatrixCSR(n, m, row_ptr, col_indices, values);

    test_with_params(row_ptr, col_indices, values, b, max_iter, tol, parallel);
}

void test1() {
    std::cout << "Test 1p begin:" << std::endl;
    std::cout << "( 4 -1 0 )   (  1.25  )   ( 15 )" << std::endl;
    std::cout << "( -1 0 4 ) x (  -10   ) = ( 10 )" << std::endl;
    std::cout << "( 0 -1 0 )   ( 2.8125 )   ( 10 )" << std::endl;
    // Пример матрицы в формате CSR
    std::vector<int> row_ptr = {0, 2, 4, 5}; // Указатели строк
    std::vector<int> col_indices = {0, 1, 0, 2, 1}; // Индексы столбцов
    std::vector<double> values = {4, -1, -1, 4, -1}; // Значения
    std::vector<double> b = {15, 10, 10};            // Вектор b

    // BiCGSTAB параметры
    int max_iter = 1000;
    double tol = 1e-6;

    test_with_params(row_ptr, col_indices, values, b, max_iter, tol, true, true);
    std::cout << std::endl;
}

void test1_1() {
    std::cout << "Test 1np begin:" << std::endl;
    std::cout << "( 4 -1 0 )   (  1.25  )   ( 15 )" << std::endl;
    std::cout << "( -1 0 4 ) x (  -10   ) = ( 10 )" << std::endl;
    std::cout << "( 0 -1 0 )   ( 2.8125 )   ( 10 )" << std::endl;
    // Пример матрицы в формате CSR
    std::vector<int> row_ptr = {0, 2, 4, 5};         // Указатели строк
    std::vector<int> col_indices = {0, 1, 0, 2, 1};  // Индексы столбцов
    std::vector<double> values = {4, -1, -1, 4, -1}; // Значения
    std::vector<double> b = {15, 10, 10};            // Вектор b

    // BiCGSTAB параметры
    int max_iter = 1000;
    double tol = 1e-6;

    test_with_params(row_ptr, col_indices, values, b, max_iter, tol, false, true);
    std::cout << std::endl;
}

void test2() {
    std::cout << "Test 2p begin:" << std::endl;
    std::cout << "( 2 -1 -1 )   ( 3 )   ( 4  )" << std::endl;
    std::cout << "( 3  4 -2 ) x ( 1 ) = ( 11 )" << std::endl;
    std::cout << "( 3 -2  4 )   ( 1 )   ( 11 )" << std::endl;
    // Пример матрицы в формате CSR
    std::vector<int> row_ptr = {0, 3, 6, 9};                        // Указатели строк
    std::vector<int> col_indices = {0, 1, 2, 0, 1, 2, 0, 1, 2};     // Индексы столбцов
    std::vector<double> values = {2, -1, -1, 3, 4, -2, 3, -2, 4};   // Значения
    std::vector<double> b = {4, 11, 11};                            // Вектор b

    // BiCGSTAB параметры
    int max_iter = 1000;
    double tol = 1e-6;

    test_with_params(row_ptr, col_indices, values, b, max_iter, tol, true, true);
    std::cout << std::endl;
}

void test2_1() {
    std::cout << "Test 2np begin:" << std::endl;
    std::cout << "( 2 -1 -1 )   ( 3 )   ( 4  )" << std::endl;
    std::cout << "( 3  4 -2 ) x ( 1 ) = ( 11 )" << std::endl;
    std::cout << "( 3 -2  4 )   ( 1 )   ( 11 )" << std::endl;
    // Пример матрицы в формате CSR
    std::vector<int> row_ptr = {0, 3, 6, 9};                        // Указатели строк
    std::vector<int> col_indices = {0, 1, 2, 0, 1, 2, 0, 1, 2};     // Индексы столбцов
    std::vector<double> values = {2, -1, -1, 3, 4, -2, 3, -2, 4};   // Значения
    std::vector<double> b = {4, 11, 11};                            // Вектор b

    // BiCGSTAB параметры
    int max_iter = 1000;
    double tol = 1e-6;

    test_with_params(row_ptr, col_indices, values, b, max_iter, tol, false, true);
    std::cout << std::endl;
}

void test3() {
    std::cout << "Test 3 begin:" << std::endl;
    test_with_generate_matrix(6000, 6000, 5000, 1e-6, true);
}
void test3_not_parallel() {
    std::cout << "Test 3 Not Parallel begin:" << std::endl;
    test_with_generate_matrix(6000, 6000, 5000, 1e-6, false);
}

int main() {
    // Тесты на корректность
    test1();
    test1_1();
    test2();
    test2_1();
    // Тесты на ускорение за счет многотопочности
    // для больших размерностей
    std::cout << "\n\n" << "1 Thread:" << "\n\n";
    test3_not_parallel();
    omp_set_num_threads(2);
    std::cout << "\n\n" << "2 Threads:" << "\n\n";
    test3();
    omp_set_num_threads(4);
    std::cout << "\n\n" << "4 Threads:" << "\n\n";
    test3();
    omp_set_num_threads(8);
    std::cout << "\n\n" << "8 Threads:" << "\n\n";
    test3();
}
