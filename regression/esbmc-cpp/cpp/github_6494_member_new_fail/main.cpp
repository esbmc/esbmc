// github #6494: the same hole via a class-level allocation function
// ([expr.new]/9), not just the global replacement rules.
#include <cstddef>
#include <cassert>

static char pool[64];

struct P
{
  int v;
  static void *operator new(size_t)
  {
    return pool;
  }
  static void operator delete(void *)
  {
  }
};

int main()
{
  P *a = new P();
  a->v = 1;
  P *b = new P();
  b->v = 2;
  assert(a->v == 1); // a and b alias, so a->v is 2
  return 0;
}
