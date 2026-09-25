// Defaulted comparison operators are neither implicit nor constructors, so
// their unnamed parameter was left unbound (github #4377). For operator<=>
// the unbound operand carried a struct type into the SMT typecast and aborted
// the solver rather than producing a counterexample.
#include <compare>
#include <cassert>

struct Eq
{
  int x;
  int y;
  bool operator==(const Eq &) const = default;
};

struct Ord
{
  int x;
  auto operator<=>(const Ord &) const = default;
};

int main()
{
  Eq a{1, 2}, b{1, 2}, c{1, 3};
  assert(a == b);
  assert(!(a == c));

  Ord p{1}, q{2};
  assert(p < q);
  assert(!(q < p));
  assert(q > p);

  return 0;
}
