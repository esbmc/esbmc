/* A pointer below its object's base has a negative offset, and the relational
   operators must read it the same way __ESBMC_POINTER_OFFSET does. Comparing
   the offsets unsigned made b-1 compare above b, so reverse iteration never
   terminated. */
#include <assert.h>

int main(void)
{
  char a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  char *b = a;
  char *below = b - 1;

  assert(__ESBMC_POINTER_OFFSET(below) == -1);
  assert(!(below >= b));
  assert(below < b);
  assert(!(below > b));
  assert(below <= b);

  assert(b + 8 > b);
  assert(b + 8 >= b);
  assert(b <= b);
  assert(b >= b);

  int sum = 0;
  for (char *p = a + 7; p >= a; p--)
    sum += *p;
  assert(sum == 36);

  return 0;
}
