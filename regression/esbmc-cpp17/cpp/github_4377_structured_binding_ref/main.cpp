#include <cassert>

struct P
{
  int a;
  int b;
};

int main()
{
  P p{1, 2};
  // Reference binding: writing through the binding must mutate the
  // underlying object.
  auto &[x, y] = p;
  x = 99;
  y = 88;
  assert(p.a == 99);
  assert(p.b == 88);
  return 0;
}
