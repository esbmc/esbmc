// These names are ordinary identifiers outside libc's reserved set, so a
// program may define them. do_special_functions matched a callee's base name
// alone, so the definitions below were discarded and the builtins verified in
// their place (#6904).
//
// Verified against clang++ -std=c++17 -fsanitize=address,undefined: exits 0.
#include <cassert>
#include <cmath>

namespace mylib
{
int abs(int x)
{
  return x < 0 ? 0 - x : x + 1;
}

double fabs(double x)
{
  return 0.0;
}

bool isnan(double x)
{
  return true;
}

bool isinf(double x)
{
  return true;
}

bool isfinite(double x)
{
  return false;
}

bool isnormal(double x)
{
  return false;
}

bool signbit(double x)
{
  return true;
}
} // namespace mylib

int main()
{
  assert(mylib::abs(5) == 6);
  assert(mylib::abs(-5) == 5);
  assert(mylib::fabs(-2.5) == 0.0);
  assert(mylib::isnan(1.0) && mylib::isinf(1.0) && mylib::signbit(1.0));
  assert(!mylib::isfinite(1.0) && !mylib::isnormal(1.0));

  // libc's own spellings keep the builtin lowering.
  const double x = 1.0;
  assert(std::abs(-7) == 7);
  assert(std::fabs(-2.5) == 2.5);
  assert(!std::isnan(x) && !std::isinf(x) && std::isfinite(x));
  assert(std::isnormal(x) && !std::signbit(x));
  return 0;
}
