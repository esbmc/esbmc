#include <python-frontend/numpy/ndarray_descriptor.h>

#include <limits>
#include <stdexcept>

namespace
{
std::vector<long long> row_major_strides(const std::vector<long long> &shape)
{
  std::vector<long long> strides(shape.size(), 0);
  long long acc = 1;
  for (std::size_t i = shape.size(); i-- > 0;)
  {
    strides[i] = acc;
    acc *= shape[i];
  }
  return strides;
}
} // namespace

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

ndarray_descriptor::ndarray_descriptor(
  std::vector<long long> shape,
  std::string dtype,
  std::size_t buffer_id,
  long long offset)
  : shape_(std::move(shape)),
    strides_(row_major_strides(shape_)),
    offset_(offset),
    dtype_(std::move(dtype)),
    buffer_id_(buffer_id)
{
  capacity_ = 1;
  for (long long dim : shape_)
    capacity_ *= dim;
}

void ndarray_descriptor::validate() const
{
  if (rank() > max_rank)
    throw std::runtime_error(
      "ValueError: ESBMC does not support arrays with more than " +
      std::to_string(max_rank) + " dimensions");

  validate_ndarray_shape(shape_);

  if (offset_ < 0 || offset_ > capacity_)
    throw std::runtime_error("ValueError: invalid ndarray offset");
}
