/* A byte-wise walk over a struct's object representation, bounded by the
 * address of another member or element, has a constant trip count. */
#include <assert.h>
struct In { short x; int y[3]; };
struct Out { char c; struct In in; int z; };
int main(void)
{
  struct Out o;
  int n1 = 0, n2 = 0, n3 = 0, n4 = 0;
  for (unsigned char *p = (unsigned char *)&o.in.y[0]; p != (unsigned char *)&o.z; ++p) ++n1;
  for (char *p = (char *)&o.in.y[2]; p != (char *)&o.in.x; --p) ++n2;
  for (char *p = (char *)&o; p < (char *)&o.in.y[1]; ++p) ++n3;
  for (char *p = (char *)&o.c; p != (char *)&o.in; ++p) ++n4;
  assert(n1 == (int)((char *)&o.z - (char *)&o.in.y[0]));
  assert(n2 == (int)((char *)&o.in.y[2] - (char *)&o.in.x));
  assert(n3 == (int)((char *)&o.in.y[1] - (char *)&o));
  assert(n4 == (int)((char *)&o.in - (char *)&o.c));
  assert(n1 == 12 && n2 == 12 && n3 == 12 && n4 == 5);
  return 0;
}
