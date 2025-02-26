#include <cassert>

template <typename>
struct a;
struct b;
template <typename>
struct a
{
  b *c;
};
struct b : a<int>
{
};

int main()
{
  b d;
  d.c = 0;
  assert(d.c != 0); // fail
  return 0;
}
