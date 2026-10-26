#pragma once

#include <vector>

/// Validates a shape vector on its own, ahead of building the backing
/// buffer (e.g. for `np.zeros`/`np.ones`/`np.full` call sites). Throws
/// std::runtime_error with a NumPy-compatible ValueError message when a
/// dimension is negative or the element count overflows.
void validate_ndarray_shape(const std::vector<long long> &shape);
