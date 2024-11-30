#include "bicgstab.h"

#include <iostream>
#include <cmath>
#include <omp.h>

namespace BiCGSTAB {

BiCGSTAB::BiCGSTAB(const bool usingParallel)
    : usingParallel(usingParallel)
{}

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
double bicgstab_parallel(const std::vector<int>& row_ptr, const std::vector<int>& col_indices,
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
    double last_err = -1;

    for (int iter = 0; iter < max_iter; ++iter) {
        rho_new = parallelDotProduct(r_hat, r);
        if (fabs(rho_old) < 1e-15) { // Защита от деления на 0
            std::cout << "Zero case." << std::endl;
            std::cout << "Iter: " << iter << std::endl;
            return last_err;
        }

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
        if (r_norm < tol * b_norm) {
            std::cout << "Iter: " << iter << std::endl;
            return r_norm / b_norm;
        }
        last_err = r_norm / b_norm;
    }
    std::cout << "Iter max." << std::endl;
    return last_err;
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
double bicgstab(const std::vector<int>& row_ptr, const std::vector<int>& col_indices,
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
    double last_err = -1;

    for (int iter = 0; iter < max_iter; ++iter) {
        rho_new = dotProduct(r_hat, r);
        if (fabs(rho_old) < 1e-15) { // Защита от деления на 0
            std::cout << "Iter: " << iter << std::endl;
            return last_err;
        }

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
            std::cout << "Iter: " << iter << std::endl;
            return r_norm / b_norm;
        }
        last_err = r_norm / b_norm;
    }
    std::cout << "Iter max." << std::endl;
    return last_err;
}

double BiCGSTAB::run(const std::vector<int>& row_ptr, const std::vector<int>& col_indices,
                   const std::vector<double>& values, const std::vector<double>& b,
                   std::vector<double>& x, int max_iter, double tol) {
    return usingParallel ?
               bicgstab_parallel(row_ptr, col_indices, values, b, x, max_iter, tol) :
               bicgstab(row_ptr, col_indices, values, b, x, max_iter, tol);
}

}
