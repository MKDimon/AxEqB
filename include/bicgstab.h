#ifndef BICGSTAB_H
#define BICGSTAB_H

#include <vector>

namespace BiCGSTAB {

class BiCGSTAB
{
public:
    explicit BiCGSTAB(const bool usingParallel = false);
    bool run(const std::vector<int>& row_ptr, const std::vector<int>& col_indices,
             const std::vector<double>& values, const std::vector<double>& b,
             std::vector<double>& x, int max_iter, double tol);

private:
    const bool usingParallel;
};
}

#endif // BICGSTAB_H
