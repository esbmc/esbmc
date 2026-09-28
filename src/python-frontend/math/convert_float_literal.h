#ifndef CPROVER_ANSI_C_CONVERT_FLOAT_LITERAL_H
#define CPROVER_ANSI_C_CONVERT_FLOAT_LITERAL_H

#include <optional>
#include <string>
#include <util/arith/ieee_float.h>
#include <util/irep/expr.h>

void convert_float_literal(const std::string &src, exprt &dest);

/// The non-finite double @p spelling names, or empty when it names none.
/// Accepts both Python's float() spellings ("inf", "+infinity", "-inf", "nan")
/// and the parser's non-finite literal tags, which are a subset (#7545).
std::optional<ieee_floatt>
nonfinite_float_from_spelling(const std::string &spelling);

#endif
