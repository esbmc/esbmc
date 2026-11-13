#pragma once

#include <optional>

/// round(value) for a finite double, ties to even as CPython rounds.
double round_ties_to_even(double value);

/// round(value, ndigits) on the exact binary value of value, as CPython
/// rounds it. Answers nullopt where CPython raises OverflowError.
std::optional<double> round_to_ndigits_ties_even(double value, int ndigits);
