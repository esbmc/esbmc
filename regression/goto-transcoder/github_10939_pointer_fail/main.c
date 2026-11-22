/* A union read over a pointer object at a dynamic zero offset is refused:
 * the solver cannot bitcast a pointer to an aggregate. */
#include <stddef.h>
union chunk { unsigned char bytes[sizeof(int *)]; };
size_t nondet_size(void);
int main(void)
{
  int y;
  int *q = &y;
  size_t off = nondet_size();
  __CPROVER_assume(off == 0);
  union chunk *p = (union chunk *)((unsigned char *)&q + off);
  union chunk a = *p;
  return a.bytes[0];
}
