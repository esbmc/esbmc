// Overloading the name `abs` for a class type used to abort with an internal
// assertion: the call was rewritten to an `abs` node whatever the argument
// type, and that lowers to `(x >= 0) ? x : -x`, which is ill-typed for a class
// and tripped if2t's type assertion. The rewrite is now limited to arithmetic
// arguments, so a user overload stays an ordinary call.
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
