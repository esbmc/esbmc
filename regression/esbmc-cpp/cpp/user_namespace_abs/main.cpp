// `abs` inside a user namespace is an ordinary identifier, not libc's. It used
// to be rewritten to the builtin `(x >= 0) ? x : -x` on the strength of its
// name alone, so ESBMC verified the builtin and not this code (#6904).
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
} // namespace mylib

int main()
{
  assert(mylib::abs(5) == 6);
  assert(mylib::abs(-5) == 5);
  assert(mylib::fabs(-2.5) == 0.0);

  // libc's own spellings keep the builtin lowering.
  assert(std::abs(-7) == 7);
  assert(std::fabs(-2.5) == 2.5);
  return 0;
}
