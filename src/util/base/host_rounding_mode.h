#pragma once

#include <cfenv>

/* RAII guard that runs a scope under one host FPU rounding mode and restores
 * the previous mode on every exit path.
 *
 * ESBMC's process-wide default is FE_TONEAREST (see main()). The one library
 * that needs something else is gaol, which ibex uses for interval arithmetic
 * and which requires FE_UPWARD; code that calls into it takes a guard for the
 * duration instead of leaving the whole process rounding upward. */
class host_rounding_mode
{
  int saved;

public:
  explicit host_rounding_mode(int mode) : saved(std::fegetround())
  {
    if (saved == mode)
      saved = -1;
    else
      std::fesetround(mode);
  }

  ~host_rounding_mode()
  {
    if (saved >= 0)
      std::fesetround(saved);
  }

  host_rounding_mode(const host_rounding_mode &) = delete;
  host_rounding_mode &operator=(const host_rounding_mode &) = delete;
};
