// Negative twin: the operands really are compared, so an equality that is
// false must be reported rather than passing vacuously.
#include <cassert>

struct P
{
  int v;
  friend bool operator==(P, P) = default;
};

int main()
{
  P a{1}, b{2};
  assert(a == b);
  return 0;
}
