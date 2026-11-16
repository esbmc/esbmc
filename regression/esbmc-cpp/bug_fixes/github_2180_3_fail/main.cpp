// #2180: a data member of empty class type occupies one byte, so writing
// that byte stays inside the enclosing object.
#include <cassert>
#include <cstring>

class empty
{
};

struct holder
{
  int c;
  empty d;
};

int main()
{
  holder e;
  e.c = 0;
  char src = 7;
  memmove(&e.d, &src, 1);
  char *p = (char *)&e.d;
  assert(*p == 0);
  return 0;
}
