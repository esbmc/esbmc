#pragma once

#include <cstddef>
#include <string>
#include <vector>

/// Canonical bounded ndarray descriptor (numpy-architecture-decisions.md).
///
/// Frontend-side bookkeeping for a numpy array's logical metadata --
/// buffer/capacity/shape/strides/offset/rank/dtype/buffer_id -- built from an
/// already-modelled array expression (an irep2 array_typet, still the sole
/// runtime representation). Consumers migrate onto this incrementally; it
/// does not replace the legacy nested-array_typet layout by itself.
class ndarray_descriptor
{
public:
  static constexpr std::size_t max_rank = 8;

  ndarray_descriptor(
    std::vector<long long> shape,
    std::string dtype,
    std::size_t buffer_id,
    long long offset = 0);

  std::size_t rank() const
  {
    return shape_.size();
  }
  const std::vector<long long> &shape() const
  {
    return shape_;
  }
  const std::vector<long long> &strides() const
  {
    return strides_;
  }
  long long capacity() const
  {
    return capacity_;
  }
  long long offset() const
  {
    return offset_;
  }
  const std::string &dtype() const
  {
    return dtype_;
  }
  std::size_t buffer_id() const
  {
    return buffer_id_;
  }

  /// Throws std::runtime_error with a NumPy-shaped diagnostic when an
  /// invariant is violated: rank <= max_rank, every logical dimension is
  /// non-negative, the shape's element count doesn't overflow, and the
  /// offset lies within the buffer.
  void validate() const;

private:
  std::vector<long long> shape_;
  std::vector<long long> strides_;
  long long capacity_;
  long long offset_;
  std::string dtype_;
  std::size_t buffer_id_;
};

/// Validates a shape vector on its own, ahead of building the backing
/// buffer (e.g. for `np.zeros`/`np.ones`/`np.full` call sites). Throws
/// std::runtime_error with a NumPy-compatible ValueError message when a
/// dimension is negative or the element count overflows.
void validate_ndarray_shape(const std::vector<long long> &shape);
