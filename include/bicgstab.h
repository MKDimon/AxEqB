#ifndef BICGSTAB_H
#define BICGSTAB_H

#include <vector>

namespace BiCGSTAB {

class BiCGSTAB
{
public:
    explicit BiCGSTAB(const bool usingParallel = false);
    double run(const std::vector<int>& row_ptr, const std::vector<int>& col_indices,
             const std::vector<double>& values, const std::vector<double>& b,
             std::vector<double>& x, int max_iter, double tol);

private:
    double last_error = -1;
    int last_amount_of_iterations = -1;
    const bool using_parallel;

    double bicgstab(const std::vector<int>& row_ptr, const std::vector<int>& col_indices,
        const std::vector<double>& values, const std::vector<double>& b,
        std::vector<double>& x, int max_iter, double tol);
    double bicgstab_parallel(const std::vector<int>& row_ptr, const std::vector<int>& col_indices,
        const std::vector<double>& values, const std::vector<double>& b,
        std::vector<double>& x, int max_iter, double tol);
    void update_return_params(const double last_error, const double last_amount_of_iterations);
};
}

#endif // BICGSTAB_H
