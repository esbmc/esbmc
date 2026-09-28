#pragma once

#include <string_view>
#include <util/base/prefix.h>

/// Which bit of the operand a __builtin_clz*/ctz*/ffs* call reports.
enum class bit_scan_endt
{
  none,
  leading,
  trailing,
  /// ffs: the one-based index of the least-significant set bit, and 0 for a
  /// zero operand -- defined there, unlike the other two.
  first_set
};

/// Recognise the __builtin_clz*/__builtin_ctz* family from a symbol name such
/// as "c:@F@__builtin_clzll". The type-generic `g` spellings take an optional
/// second argument, the result to use when the operand is zero -- which is what
/// makes them defined there, unlike the rest of the family (GCC, "Bit Operation
/// Builtins").
///
/// The width suffix is matched against a closed set rather than by prefix: a
/// loose "__builtin_clz" test would also capture unrelated builtins that happen
/// to start the same way, which is how the two-argument clzg once reached a
/// one-argument handler (#4606).
inline bit_scan_endt bit_scan_builtin(std::string_view symname)
{
  constexpr std::string_view prefix = "c:@F@__builtin_";
  if (!has_prefix(symname, prefix))
    return bit_scan_endt::none;
  symname.remove_prefix(prefix.size());

  bit_scan_endt end = bit_scan_endt::none;
  if (has_prefix(symname, "clz"))
    end = bit_scan_endt::leading;
  else if (has_prefix(symname, "ctz"))
    end = bit_scan_endt::trailing;
  else if (has_prefix(symname, "ffs"))
    end = bit_scan_endt::first_set;
  else
    return bit_scan_endt::none;
  symname.remove_prefix(3);

  for (const std::string_view width : {"", "l", "ll"})
    if (symname == width)
      return end;

  // Only clz/ctz have the 16-bit and type-generic spellings.
  if (end == bit_scan_endt::first_set)
    return bit_scan_endt::none;

  for (const std::string_view width : {"s", "g"})
    if (symname == width)
      return end;
  return bit_scan_endt::none;
}
