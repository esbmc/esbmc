// Negative direction for github #4377: a defaulted assignment must copy the
// real operand, so a wrong expectation has to be reported as a violation
// rather than satisfied by a nondet value.
#include <cassert>

struct Copyable
{
  int x;
  Copyable &operator=(const Copyable &) = default;
};

int main()
{
  Copyable a{1}, b{30};
  a = b;
  assert(a.x == 1); // must fail: a.x is 30 after the assignment
  return 0;
}
