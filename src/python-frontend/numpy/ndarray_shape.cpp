#include <python-frontend/numpy/ndarray_shape.h>

#include <limits>
#include <stdexcept>

void validate_ndarray_shape(const std::vector<long long> &shape)
{
  for (long long dim : shape)
    if (dim < 0)
      throw std::runtime_error(
        "ValueError: negative dimensions are not allowed");

  long long product = 1;
  for (long long dim : shape)
  {
    if (dim != 0 && product > std::numeric_limits<long long>::max() / dim)
      throw std::runtime_error(
        "ValueError: array size overflows during creation");
    product *= dim;
  }
}
