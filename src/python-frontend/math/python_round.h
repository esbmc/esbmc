#pragma once

#include <optional>

/// round(value) for a finite double, ties to even as CPython rounds.
double round_ties_to_even(double value);

/// round(value, ndigits) on the exact binary value of value, as CPython
/// rounds it. Answers nullopt where CPython raises OverflowError.
std::optional<double> round_to_ndigits_ties_even(double value, int ndigits);

/// round(value, ndigits) of an int, rounding half to even at 10**-ndigits.
/// Answers nullopt when the result does not fit in long long.
std::optional<long long> round_int_to_ndigits(long long value, int ndigits);
