#pragma once

#include <cfenv>

// RAII guard that pins the host floating-point rounding mode to FE_TONEAREST
// for the duration of a scope and restores the previous mode on exit.
//
// The surrounding pipeline can leave the host in a non-default rounding mode
// (the GOTO/symex float model drives fesetround via __ESBMC_rounding_mode).
// Host printf/snprintf and std::round/ostream conversions honour that mode,
// so under FE_UPWARD, e.g. "%.2f" of -1.5 renders "-1.49" instead of "-1.50"
// (observed on the Linux CI, where the host was left rounding upward). CPython
// formats with round-half-to-even, i.e. FE_TONEAREST, so any compile-time
// float-to-decimal fold must pin that mode to stay correct regardless of the
// host's current mode. The guard restores on every return path.
struct round_to_nearest_guard
{
  int saved;
  round_to_nearest_guard() : saved(std::fegetround())
  {
    if (saved != FE_TONEAREST)
      std::fesetround(FE_TONEAREST);
  }
  ~round_to_nearest_guard()
  {
    if (saved >= 0 && saved != FE_TONEAREST)
      std::fesetround(saved);
  }
};
