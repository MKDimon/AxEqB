#include "bicgstab.h"

#include <iostream>
#include <cmath>
#include <omp.h>

namespace BiCGSTAB {

BiCGSTAB::BiCGSTAB(const bool usingParallel)
    : using_parallel(usingParallel)
{}

// Параллельное умножение матрицы CSR на вектор
void sparse_matrix_vector_parallel_multiply_CSR(const std::vector<int>& row_ptr,
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
double parallel_dot_product(const std::vector<double>& v1, const std::vector<double>& v2) {
    double result = 0.0;
#pragma omp parallel for reduction(+:result)
    for (int i = 0; i < v1.size(); ++i) {
        result += v1[i] * v2[i];
    }
    return result;
}

// Это должно вызываться 
void BiCGSTAB::update_return_params(const double last_error, const double last_amount_of_iterations) {
    this->last_error = last_error;
    this->last_amount_of_iterations = last_amount_of_iterations;
}


// BiCGSTAB с параллельной итерацией
double BiCGSTAB::bicgstab_parallel(const std::vector<int>& row_ptr, const std::vector<int>& col_indices,
                       const std::vector<double>& values, const std::vector<double>& b,
                       std::vector<double>& x, int max_iter, double tol) {
    int n = b.size();
    std::vector<double> r(n), r_hat(n), v(n), p(n), s(n), t(n);

    // Инициализация
    sparse_matrix_vector_parallel_multiply_CSR(row_ptr, col_indices, values, x, r);
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        r[i] = b[i] - r[i];
    }
    r_hat = r;
    double rho_old = 1.0, alpha = 1.0, omega = 1.0;
    double rho_new, beta;

    double b_norm = sqrt(parallel_dot_product(b, b));
    if (b_norm == 0.0) b_norm = 1.0;
    this->last_error = -1;

    for (int iter = 0; iter < max_iter; ++iter) {
        rho_new = parallel_dot_product(r_hat, r);
        if (fabs(rho_old) < 1e-15) { // Защита от деления на 0
            std::cout << "Zero case." << std::endl;
            std::cout << "Iter: " << iter << std::endl;
            this->update_return_params(this->last_error, iter);
            return this->last_error;
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
        sparse_matrix_vector_parallel_multiply_CSR(row_ptr, col_indices, values, p, v);
        alpha = rho_new / parallel_dot_product(r_hat, v);

#pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            s[i] = r[i] - alpha * v[i];
        }

        // Вычисление t = A * s
        sparse_matrix_vector_parallel_multiply_CSR(row_ptr, col_indices, values, s, t);
        omega = parallel_dot_product(t, s) / parallel_dot_product(t, t);

#pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            x[i] += alpha * p[i] + omega * s[i];
            r[i] = s[i] - omega * t[i];
        }

        rho_old = rho_new;

        double r_norm = sqrt(parallel_dot_product(r, r));
        if (r_norm < tol * b_norm) {
            std::cout << "Iter: " << iter << std::endl;
            this->update_return_params(r_norm / b_norm, iter);
            return r_norm / b_norm;
        }
        this->last_error = r_norm / b_norm;
    }
    std::cout << "Iter max." << std::endl;
    this->update_return_params(this->last_error, max_iter);
    return this->last_error;
}

// Умножение матрицы CSR на вектор
void sparse_matrix_vector_multiply_CSR(const std::vector<int>& row_ptr,
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

// Непараллельное скалярное произведение
double dot_product(const std::vector<double>& v1, const std::vector<double>& v2) {
    double result = 0.0;
    for (int i = 0; i < v1.size(); ++i) {
        result += v1[i] * v2[i];
    }
    return result;
}

// BiCGSTAB без распараллеливания
double BiCGSTAB::bicgstab(const std::vector<int>& row_ptr, const std::vector<int>& col_indices,
              const std::vector<double>& values, const std::vector<double>& b,
              std::vector<double>& x, int max_iter, double tol) {
    int n = b.size();
    std::vector<double> r(n), r_hat(n), v(n), p(n), s(n), t(n);

    // Инициализация
    sparse_matrix_vector_multiply_CSR(row_ptr, col_indices, values, x, r);
    for (int i = 0; i < n; ++i) {
        r[i] = b[i] - r[i];
    }
    r_hat = r;
    double rho_old = 1.0, alpha = 1.0, omega = 1.0;
    double rho_new, beta;

    double b_norm = sqrt(dot_product(b, b));
    if (b_norm == 0.0) b_norm = 1.0;
    this->last_error = -1;

    for (int iter = 0; iter < max_iter; ++iter) {
        rho_new = dot_product(r_hat, r);
        if (fabs(rho_old) < 1e-15) { // Защита от деления на 0
            std::cout << "Iter: " << iter << std::endl;
            this->update_return_params(this->last_error, iter);
            return this->last_error;
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
        sparse_matrix_vector_multiply_CSR(row_ptr, col_indices, values, p, v);
        alpha = rho_new / dot_product(r_hat, v);

        for (int i = 0; i < n; ++i) {
            s[i] = r[i] - alpha * v[i];
        }

        // Вычисление t = A * s
        sparse_matrix_vector_multiply_CSR(row_ptr, col_indices, values, s, t);
        omega = dot_product(t, s) / dot_product(t, t);

        for (int i = 0; i < n; ++i) {
            x[i] += alpha * p[i] + omega * s[i];
            r[i] = s[i] - omega * t[i];
        }

        rho_old = rho_new;

        double r_norm = sqrt(dot_product(r, r));
        if (r_norm / b_norm < tol) {
            std::cout << "Iter: " << iter << std::endl;
            this->update_return_params(r_norm / b_norm, iter);
            return r_norm / b_norm;
        }
        this->last_error = r_norm / b_norm;
    }
    std::cout << "Iter max." << std::endl;
    this->update_return_params(this->last_error, max_iter);
    return this->last_error;
}

double BiCGSTAB::run(const std::vector<int>& row_ptr, const std::vector<int>& col_indices,
                   const std::vector<double>& values, const std::vector<double>& b,
                   std::vector<double>& x, int max_iter, double tol) {
    return using_parallel ?
               bicgstab_parallel(row_ptr, col_indices, values, b, x, max_iter, tol) :
               bicgstab(row_ptr, col_indices, values, b, x, max_iter, tol);
}

}
