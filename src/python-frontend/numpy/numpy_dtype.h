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

/// Whether the consumer @p name, a numpy function or an array attribute, is
/// modelled over elements of @p kind. Numbers are; a string or complex element
/// reads as nothing else, so anything not listed for it must reject.
bool numpy_consumer_is_modelled(
  numpy_dtypet::kindt kind,
  const std::string &name);

/// "string" or "complex": what a rejection calls an array of @p kind.
std::string numpy_kind_name(numpy_dtypet::kindt kind);

/// The dtype a normalized scalar name denotes.
std::optional<numpy_dtypet> numpy_dtype_from_name(const std::string &name);

/// The dtype of a storage element; nullopt when @p element is not one.
std::optional<numpy_dtypet> numpy_dtype_of_element(const typet &element);

/// An operand of a NumPy operation. A Python number is weak (NEP 50): it takes
/// the other operand's dtype instead of taking part by its own width.
struct numpy_operandt
{
  numpy_dtypet dtype;
  bool weak;
};

/// The result dtype of an operation over @p a and @p b.
numpy_dtypet numpy_promote(const numpy_operandt &a, const numpy_operandt &b);
