#include <assert.h>

int nondet_int(void);
int x, y;

int main()
{
  int *p = nondet_int() ? &x : &y;
  void *v = &p;
  double d = *(double *)v;
  double e = *(double *)&p;
  assert(d == e || d != d);
  return 0;
}
