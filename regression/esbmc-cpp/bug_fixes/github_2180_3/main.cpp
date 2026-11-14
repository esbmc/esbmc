// #2180: a data member of empty class type occupies one byte, so writing
// that byte stays inside the enclosing object.
#include <cassert>
#include <cstddef>
#include <cstring>

class empty
{
};

struct alignas(8) empty8
{
};

struct holder
{
  int c;
  empty d;
};

struct leading
{
  empty d;
  int c;
};

struct aligned
{
  empty8 e;
  int x;
};

static holder g;

int main()
{
  holder e;
  e.c = 0;
  char src = 7;
  memmove(&e.d, &src, 1);
  char *p = (char *)&e.d;
  assert(*p == 7);
  *p = 9;
  assert(e.c == 0);

  leading l{{}, 5};
  assert(l.c == 5);
  holder z{};
  assert(z.c == 0 && g.c == 0);

  aligned a{{}, 3};
  int v;
  memcpy(&v, (char *)&a + offsetof(aligned, x), sizeof v);
  assert(v == 3);
  return 0;
}
