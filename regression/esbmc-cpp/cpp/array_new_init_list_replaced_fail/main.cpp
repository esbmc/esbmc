// A braced list initialises storage from a replaced operator new[].
#include <cassert>
#include <cstddef>
#include <new>
static char pool[64];
static int used = 0;
void *operator new[](std::size_t n)
{
  void *r = pool + used;
  used += (int)n;
  return r;
}
void operator delete[](void *) noexcept
{
}
int main()
{
  int *p = new int[2]{7, 8};
  assert(p[1] != 8);
  return 0;
}
