#include <cassert>

struct P
{
  int a;
  int b;
};

int main()
{
  P p{1, 2};
  auto [x, y] = p;
  // The bindings hold (1, 2); the assertion below must fail.
  assert(x == 9);
  return 0;
}
