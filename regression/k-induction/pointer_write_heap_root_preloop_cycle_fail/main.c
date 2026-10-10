// As pointer_write_heap_root_preloop_alloc_fail, with the trace reaching g a
// second time through v, which g's result was assigned before the loop.
#include <stdlib.h>
#include <assert.h>
extern int nondet_int(void);
long g(long a)
{
  long x;
  if (a)
    x = a;
  else
    x = (long)malloc(sizeof(int));
  return x;
}
int *h(long a) { return (int *)g(a); }
int main()
{
  long v;
  v = g(0);
  *(int *)v = 0;
  while (nondet_int())
  {
    int *r;
    r = h(v);
    *r = *r + 1;
    assert(*(int *)v < 3);
  }
  return 0;
}
