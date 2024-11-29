#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <iomanip>

// Умножение матрицы CSR на вектор
void sparseMatrixVectorParallelMultiplyCSR(const std::vector<int>& row_ptr,
                                   const std::vector<int>& col_indices,
                                   const std::vector<double>& values,
                                   const std::vector<double>& vec,
                                   std::vector<double>& result) {
    int rows = row_ptr.size() - 1;
    result.assign(rows, 0.0);

#pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        double sum = 0.0;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            sum += values[j] * vec[col_indices[j]];
        }
        result[i] = sum;
    }
}

// Параллельное скалярное произведение
double parallelDotProduct(const std::vector<double>& v1, const std::vector<double>& v2) {
    double result = 0.0;
#pragma omp parallel for reduction(+:result)
    for (int i = 0; i < v1.size(); ++i) {
        result += v1[i] * v2[i];
    }
    return result;
}

// BiCGSTAB с параллельной итерацией
bool bicgstab_parallel(const std::vector<int>& row_ptr, const std::vector<int>& col_indices,
                       const std::vector<double>& values, const std::vector<double>& b,
                       std::vector<double>& x, int max_iter, double tol) {
    int n = b.size();
    std::vector<double> r(n), r_hat(n), v(n), p(n), s(n), t(n);

    // Инициализация
    sparseMatrixVectorParallelMultiplyCSR(row_ptr, col_indices, values, x, r);
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        r[i] = b[i] - r[i];
    }
    r_hat = r;
    double rho_old = 1.0, alpha = 1.0, omega = 1.0;
    double rho_new, beta;

    double b_norm = sqrt(parallelDotProduct(b, b));
    if (b_norm == 0.0) b_norm = 1.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        rho_new = parallelDotProduct(r_hat, r);
        if (fabs(rho_new) < 1e-15) return false; // Защита от деления на 0

        if (iter == 0) {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                p[i] = r[i];
            }
        } else {
            beta = (rho_new / rho_old) * (alpha / omega);
#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                p[i] = r[i] + beta * (p[i] - omega * v[i]);
            }
        }

        // Вычисление v = A * p
        sparseMatrixVectorParallelMultiplyCSR(row_ptr, col_indices, values, p, v);
        alpha = rho_new / parallelDotProduct(r_hat, v);

#pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            s[i] = r[i] - alpha * v[i];
        }

        double s_norm = sqrt(parallelDotProduct(s, s));
        if (s_norm / b_norm < tol) {
#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                x[i] += alpha * p[i];
            }
            return true;
        }

        // Вычисление t = A * s
        sparseMatrixVectorParallelMultiplyCSR(row_ptr, col_indices, values, s, t);
        omega = parallelDotProduct(t, s) / parallelDotProduct(t, t);

#pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            x[i] += alpha * p[i] + omega * s[i];
            r[i] = s[i] - omega * t[i];
        }

        rho_old = rho_new;

        double r_norm = sqrt(parallelDotProduct(r, r));
        if (r_norm / b_norm < tol) {
            return true;
        }
    }
    return false;
}
// Умножение матрицы CSR на вектор
void sparseMatrixVectorMultiplyCSR(const std::vector<int>& row_ptr,
                                   const std::vector<int>& col_indices,
                                   const std::vector<double>& values,
                                   const std::vector<double>& vec,
                                   std::vector<double>& result) {
    int rows = row_ptr.size() - 1;
    result.assign(rows, 0.0);

    for (int i = 0; i < rows; ++i) {
        double sum = 0.0;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            sum += values[j] * vec[col_indices[j]];
        }
        result[i] = sum;
    }
}

// Параллельное скалярное произведение
double dotProduct(const std::vector<double>& v1, const std::vector<double>& v2) {
    double result = 0.0;
    for (int i = 0; i < v1.size(); ++i) {
        result += v1[i] * v2[i];
    }
    return result;
}

// BiCGSTAB с параллельной итерацией
bool bicgstab(const std::vector<int>& row_ptr, const std::vector<int>& col_indices,
                       const std::vector<double>& values, const std::vector<double>& b,
                       std::vector<double>& x, int max_iter, double tol) {
    int n = b.size();
    std::vector<double> r(n), r_hat(n), v(n), p(n), s(n), t(n);

    // Инициализация
    sparseMatrixVectorMultiplyCSR(row_ptr, col_indices, values, x, r);
    for (int i = 0; i < n; ++i) {
        r[i] = b[i] - r[i];
    }
    r_hat = r;
    double rho_old = 1.0, alpha = 1.0, omega = 1.0;
    double rho_new, beta;

    double b_norm = sqrt(dotProduct(b, b));
    if (b_norm == 0.0) b_norm = 1.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        rho_new = dotProduct(r_hat, r);
        if (fabs(rho_new) < 1e-15) return false; // Защита от деления на 0

        if (iter == 0) {
            for (int i = 0; i < n; ++i) {
                p[i] = r[i];
            }
        } else {
            beta = (rho_new / rho_old) * (alpha / omega);
            for (int i = 0; i < n; ++i) {
                p[i] = r[i] + beta * (p[i] - omega * v[i]);
            }
        }

        // Вычисление v = A * p
        sparseMatrixVectorMultiplyCSR(row_ptr, col_indices, values, p, v);
        alpha = rho_new / dotProduct(r_hat, v);

        for (int i = 0; i < n; ++i) {
            s[i] = r[i] - alpha * v[i];
        }

        double s_norm = sqrt(dotProduct(s, s));
        if (s_norm / b_norm < tol) {
            for (int i = 0; i < n; ++i) {
                x[i] += alpha * p[i];
            }
            return true;
        }

        // Вычисление t = A * s
        sparseMatrixVectorMultiplyCSR(row_ptr, col_indices, values, s, t);
        omega = dotProduct(t, s) / dotProduct(t, t);

        for (int i = 0; i < n; ++i) {
            x[i] += alpha * p[i] + omega * s[i];
            r[i] = s[i] - omega * t[i];
        }

        rho_old = rho_new;

        double r_norm = sqrt(dotProduct(r, r));
        if (r_norm / b_norm < tol) {
            return true;
        }
    }
    return false;
}

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
                      const int max_iter, const double tol, bool parallel) {

    std::vector<double> x(b.size(), 0); // Начальное приближение
    // Старт замера
    std::cout << (parallel ? "Parallel calc:" : "Not parallel calc") << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    if (parallel) {
        if (bicgstab_parallel(row_ptr, col_indices, values, b, x, max_iter, tol)) {
            // Конец замера
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            std::cout << "Time running: " << elapsed_seconds.count() << " sec" << std::endl;
            std::cout << "Solution found: ";
            for (double val : x) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "Method did not converge in parallel try." << std::endl;
        }
    } else {
        if (bicgstab(row_ptr, col_indices, values, b, x, max_iter, tol)) {
            // Конец замера
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            std::cout << "Time running: " << elapsed_seconds.count() << " sec" << std::endl;
            // Вывод решения
            std::cout << "Solution found: ";
            for (double val : x) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "Method did not converge in not parallel try." << std::endl;
        }
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
    std::vector<int> row_ptr;    // Указатели строк
    std::vector<int> col_indices;// Индексы столбцов
    std::vector<double> values;  // Значения
    std::vector<double> b(n, 1);       // Вектор b

    generateSparseMatrixCSR(n, m, row_ptr, col_indices, values);
    //printMatrixCSR(n, m, row_ptr, col_indices, values);

    test_with_params(row_ptr, col_indices, values, b, max_iter, tol, parallel);
}

void test1() {
    std::cout << "Test 1 begin:" << std::endl;
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

    test_with_params(row_ptr, col_indices, values, b, max_iter, tol, true);
}

void test1_1() {
    std::cout << "Test 1 begin:" << std::endl;
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

    test_with_params(row_ptr, col_indices, values, b, max_iter, tol, false);
}

void test2() {
    std::cout << "Test 2 begin:" << std::endl;
    std::cout << "( 2 -1 -1 )   ( 3 )   ( 4  )" << std::endl;
    std::cout << "( 3  4 -2 ) x ( 1 ) = ( 11 )" << std::endl;
    std::cout << "( 3 -2  4 )   ( 1 )   ( 11 )" << std::endl;
    // Пример матрицы в формате CSR
    std::vector<int> row_ptr = {0, 3, 6, 9}; // Указатели строк
    std::vector<int> col_indices = {0, 1, 2, 0, 1, 2, 0, 1, 2}; // Индексы столбцов
    std::vector<double> values = {2, -1, -1, 3, 4, -2, 3, -2, 4}; // Значения
    std::vector<double> b = {4, 11, 11};            // Вектор b

    // BiCGSTAB параметры
    int max_iter = 1000;
    double tol = 1e-6;

    test_with_params(row_ptr, col_indices, values, b, max_iter, tol, true);
}

void test2_1() {
    std::cout << "Test 2 begin:" << std::endl;
    std::cout << "( 2 -1 -1 )   ( 3 )   ( 4  )" << std::endl;
    std::cout << "( 3  4 -2 ) x ( 1 ) = ( 11 )" << std::endl;
    std::cout << "( 3 -2  4 )   ( 1 )   ( 11 )" << std::endl;
    // Пример матрицы в формате CSR
    std::vector<int> row_ptr = {0, 3, 6, 9}; // Указатели строк
    std::vector<int> col_indices = {0, 1, 2, 0, 1, 2, 0, 1, 2}; // Индексы столбцов
    std::vector<double> values = {2, -1, -1, 3, 4, -2, 3, -2, 4}; // Значения
    std::vector<double> b = {4, 11, 11};            // Вектор b

    // BiCGSTAB параметры
    int max_iter = 1000;
    double tol = 1e-6;

    test_with_params(row_ptr, col_indices, values, b, max_iter, tol, false);
}

void test3() {
    std::cout << "Test 3 begin:" << std::endl;
    test_with_generate_matrix(20, 5, 10000, 1e-6, true);
    test_with_generate_matrix(20, 5, 10000, 1e-6, false);
}

int main() {
    omp_set_num_threads(4);
    test1();
    test1_1();
    test2();
    test2_1();
    test3();
}
