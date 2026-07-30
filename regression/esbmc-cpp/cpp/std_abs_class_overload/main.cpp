// KNOWNBUG: overloading the name `abs` for a class type makes ESBMC abort with
// an internal assertion:
//
//   Assertion failed: (type->type_id == trueval->type->type_id),
//   function if2t, file irep2_expr.h, line 809
//
// The identical body under any other name converts and verifies fine, so the
// trigger is the name `abs` rather than the arithmetic. This is why <complex>
// does not provide std::abs(complex) -- shipping it would ship a crash.
//
// Verified against clang++ -std=c++17 -fsanitize=address,undefined: exits 0.
#include <cmath>
#include <cassert>

namespace std
{
struct Mag
{
  double v;
};

double abs(const Mag &m)
{
  return sqrt(m.v * m.v);
}
} // namespace std

int main()
{
  std::Mag m;
  m.v = -5.0;
  assert(std::abs(m) == 5.0);
  return 0;
}
