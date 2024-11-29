#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include "bicgstab.h"

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>

using Eigen::VectorXd;
using Eigen::SparseMatrix;

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
    auto err = worker.run(row_ptr, col_indices, values, b, x, max_iter, tol);
    // Конец замера
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Time running: " << elapsed_seconds.count() << " sec" << std::endl;
    std::cout << "Error: " << err << std::endl;
    if (print_answer){
        std::cout << "Solution found: ";
        for (double val : x) {
            std::cout << val << " ";
        }
    }
    std::cout << std::endl;
}

// Функция для генерации разреженной ленточной матрицы в формате CSR
void generateSparseMatrixCSR(int n, int m,
                             std::vector<int>& row_ptr,
                             std::vector<int>& col_indices,
                             std::vector<double>& values) {
    static const int MAX_VALUE = 10;
    row_ptr.resize(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        for (int j = std::max(0, i - m); j <= std::min(n - 1, i + m); ++j) {
            values.push_back(std::rand() % 10); // Заполняем значением 1.0
            col_indices.push_back(j);
        }
        row_ptr[i + 1] = values.size();
    }
}

// Конвертация CSR в Eigen::SparseMatrix
Eigen::SparseMatrix<double> convertCSRToEigen(
    int rows, int cols,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_indices,
    const std::vector<double>& values) {

    // Создаём список триплетов для заполнения
    std::vector<Eigen::Triplet<double>> triplets;

    // Перебираем строки
    for (int i = 0; i < rows; ++i) {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            triplets.emplace_back(i, col_indices[j], values[j]);
        }
    }

    // Создаём разреженную матрицу
    Eigen::SparseMatrix<double> mat(rows, cols);
    mat.setFromTriplets(triplets.begin(), triplets.end());

    return mat;
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

void test_eigen(SparseMatrix<double> A, int n = 100, int m = 100,
                int max_iter = 30, bool needPrint = false) {
    //printEigenSparseMatrix(A);
    VectorXd x(m), b(n);
    b.setOnes();

    Eigen::BiCGSTAB<SparseMatrix<double> > solver;
    solver.setMaxIterations(max_iter);
    solver.setTolerance(1e-6);
    solver.compute(A);

    auto start = std::chrono::high_resolution_clock::now();
    x = solver.solve(b);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Time running: " << elapsed_seconds.count() << " sec" << std::endl;
    std::cout << "#iterations:     " << solver.iterations() << std::endl;
    std::cout << "estimated error: " << solver.error()      << std::endl;
    if (needPrint) {
        for (auto& i: x) {
            std::cout << i << " ";
        }
        std::cout << std::endl;
    }
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
    int max_iter = 30;
    double tol = 1e-6;

    test_with_params(row_ptr, col_indices, values, b, max_iter, tol, true, true);

    std::cout << std::endl << "Eigen test:" << std::endl;

    auto A = convertCSRToEigen(3, 3, row_ptr, col_indices, values);
    VectorXd xe(3);
    VectorXd be(3);
    be[0] = 15;
    be[1] = 10;
    be[2] = 10;

    Eigen::BiCGSTAB<SparseMatrix<double> > solver;
    solver.setMaxIterations(max_iter);
    solver.setTolerance(1e-6);
    solver.compute(A);

    auto start = std::chrono::high_resolution_clock::now();
    xe = solver.solve(be);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Time: " << elapsed_seconds.count() << " sec" << std::endl;
    std::cout << "#iterations:     " << solver.iterations() << std::endl;
    std::cout << "estimated error: " << solver.error()      << std::endl;
    for (auto item: xe) {
        std::cout << item << " ";
    }
    std::cout << std::endl << std::endl;
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
    int max_iter = 30;
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
    int max_iter = 30;
    double tol = 1e-6;

    test_with_params(row_ptr, col_indices, values, b, max_iter, tol, true, true);

    std::cout << std::endl << "Eigen test:" << std::endl;

    auto A = convertCSRToEigen(3, 3, row_ptr, col_indices, values);
    VectorXd xe(3);
    VectorXd be(3);
    be[0] = 4;
    be[1] = 11;
    be[2] = 11;

    Eigen::BiCGSTAB<SparseMatrix<double> > solver;
    solver.setMaxIterations(max_iter);
    solver.setTolerance(1e-6);
    solver.compute(A);

    auto start = std::chrono::high_resolution_clock::now();
    xe = solver.solve(be);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Time: " << elapsed_seconds.count() << " sec" << std::endl;
    std::cout << "#iterations:     " << solver.iterations() << std::endl;
    std::cout << "estimated error: " << solver.error()      << std::endl;
    for (auto item: xe) {
        std::cout << item << " ";
    }
    std::cout << std::endl << std::endl;
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
    int max_iter = 30;
    double tol = 1e-6;

    test_with_params(row_ptr, col_indices, values, b, max_iter, tol, false, true);
    std::cout << std::endl;
}

void test3(const std::vector<int> row_ptr, const std::vector<int> col_indices,
           const std::vector<double> values, const std::vector<double> b,
           int n = 300, int m = 300, int max_iter = 30, bool needPrint = false) {
    std::cout << "Test 3 begin:" << std::endl;
    test_with_params(row_ptr, col_indices, values, b, max_iter, 1e-6, true, needPrint);
}
void test3_not_parallel(const std::vector<int> row_ptr, const std::vector<int> col_indices,
                        const std::vector<double> values, const std::vector<double> b,
                        int n = 300, int m = 300, int max_iter = 30, bool needPrint = false) {
    std::cout << "Test 3 Not Parallel begin:" << std::endl;
    test_with_params(row_ptr, col_indices, values, b, max_iter, 1e-6, false, needPrint);
}

int main() {
    int n = 300;
    int m = n;
    int max_iter = 40;
    // Тесты на корректность
    test1();
    test1_1();
    test2();
    test2_1();
    // Тесты на ускорение за счет многотопочности
    // для больших размерностей

    std::vector<int> row_ptr;           // Указатели строк
    std::vector<int> col_indices;       // Индексы столбцов
    std::vector<double> values;         // Значения
    std::vector<double> b(n, 1);        // Вектор b

    generateSparseMatrixCSR(n, m, row_ptr, col_indices, values);

    std::cout << "\n\n" << "1 Thread:" << "\n\n";
    test3_not_parallel(row_ptr, col_indices, values, b, n, m, max_iter);
    omp_set_num_threads(2);
    std::cout << "\n\n" << "2 Threads:" << "\n\n";
    test3(row_ptr, col_indices, values, b, n, m, max_iter);
    omp_set_num_threads(4);
    std::cout << "\n\n" << "4 Threads:" << "\n\n";
    test3(row_ptr, col_indices, values, b, n, m, max_iter);

    std::cout << "\n\n" << "Eigen test:" << "\n\n";
    test_eigen(convertCSRToEigen(n, m, row_ptr, col_indices, values), n, m, max_iter);
}
