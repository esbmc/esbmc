/* Rearranging the bounds check must not make it over-fire: every in-bounds
   byte offset still verifies, including the tight last one at
   data_sz - access_sz. */
#include <assert.h>

struct c
{
  char a;
  int b;
  long long d;
};

int main(void)
{
  struct c s;
  s.a = 1;
  s.b = 2;
  s.d = 3;

  char *p = (char *)&s;

  assert(*(char *)(p + 0) == 1);
  assert(*(int *)(p + 4) == 2);
  assert(*(long long *)(p + 8) == 3);
  assert(*(char *)(p + 15) == *((char *)&s.d + 7));

  return 0;
}
