#pragma once

#include <string_view>
#include <util/base/prefix.h>

/// Which end of the operand a __builtin_clz*/__builtin_ctz* call counts from.
enum class bit_scan_endt
{
  none,
  leading,
  trailing
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

  bit_scan_endt end;
  if (has_prefix(symname, "clz"))
    end = bit_scan_endt::leading;
  else if (has_prefix(symname, "ctz"))
    end = bit_scan_endt::trailing;
  else
    return bit_scan_endt::none;
  symname.remove_prefix(3);

  if (
    symname.empty() || symname == "l" || symname == "ll" || symname == "s" ||
    symname == "g")
    return end;
  return bit_scan_endt::none;
}
