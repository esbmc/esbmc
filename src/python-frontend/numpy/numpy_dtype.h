#pragma once

#include <util/irep/type.h>

#include <cstddef>
#include <optional>
#include <string>

/// A NumPy scalar dtype, the one description storage, casts and metadata read.
struct numpy_dtypet
{
  enum class kindt
  {
    boolean,
    signed_integer,
    unsigned_integer,
    floating,
    complex,
    string
  };

  kindt kind;
  /// Bits of the element; for a complex dtype, of both components together.
  unsigned bits;

  bool is_integer() const
  {
    return kind == kindt::signed_integer || kind == kindt::unsigned_integer;
  }

  /// The canonical name, `int8` or `complex64`.
  std::string name() const;
  /// The element type of the array's storage. Nil for a string, whose width
  /// comes from the array.
  typet element_type() const;
  /// Bytes per element, as `.itemsize` reports them.
  std::size_t itemsize() const;
};

/// The dtype a normalized scalar name denotes.
std::optional<numpy_dtypet> numpy_dtype_from_name(const std::string &name);

/// The dtype of a storage element; nullopt when @p element is not one.
std::optional<numpy_dtypet> numpy_dtype_of_element(const typet &element);
