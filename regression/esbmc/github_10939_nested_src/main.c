/* A union read at constant offset 0 over a struct whose first member is a
 * narrower struct: it spans both members, so it is rebuilt (#10939). */
#include <assert.h>
struct d { struct { unsigned char a; } s; unsigned char r[3]; };
union u { unsigned char b[4]; };
int main(void)
{
  struct d d = {{1}, {2, 3, 4}};
  union u v = *(union u *)&d;
  assert(v.b[0] == 1 && v.b[3] == 4);
  return 0;
}
