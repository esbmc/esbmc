/* A byte-wise walk over a struct's object representation, bounded by the
 * address of another member or element, has a constant trip count. */
#include <assert.h>
struct S { int a[2]; int b[2]; };
int main(void)
{
  struct S s;
  int n = 0;
  for (char *p = (char *)&s.a[0]; p != (char *)&s.a[1]; ++p)
    ++n;
  assert(n == 4);
  return 0;
}
