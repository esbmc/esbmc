// An explicitly-defaulted assignment operator is defaulted but not implicit,
// so its unnamed parameter was left unbound and the synthesised body read a
// nondet operand (github #4377).
#include <cassert>
#include <utility>

struct Copyable
{
  int x;
  int y;
  Copyable &operator=(const Copyable &) = default;
};

struct Movable
{
  int x;
  Movable(int v) : x(v)
  {
  }
  Movable(Movable &&) = default;
  Movable &operator=(Movable &&) = default;
};

int main()
{
  Copyable a{1, 2}, b{30, 40};
  a = b;
  assert(a.x == 30);
  assert(a.y == 40);

  Movable m(1), n(7);
  m = std::move(n);
  assert(m.x == 7);

  return 0;
}
