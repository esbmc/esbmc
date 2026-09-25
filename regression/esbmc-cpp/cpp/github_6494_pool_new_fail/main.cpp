// github #6494: a replaced ::operator new that hands out the same storage
// twice makes two live objects alias, so *a is clobbered by the second new.
// ESBMC used to conjure a fresh object for every new-expression and never call
// the replacement, proving this program safe -- a missed bug. g++ builds it and
// the assertion fires at runtime.
#include <cstddef>
#include <cassert>

static char pool[64];

void *operator new(size_t)
{
  return pool;
}
void operator delete(void *) noexcept
{
}

int main()
{
  int *a = new int(1);
  int *b = new int(2);
  assert(*a == 1); // a and b alias, so *a is 2
  return 0;
}
