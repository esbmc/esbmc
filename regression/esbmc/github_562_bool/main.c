#include <assert.h>
extern int nondet_int(void);
int main(void)
{
  int c = nondet_int();
  double u = (c != 0);
  assert((int)u == (c != 0 ? 1 : 0));
  return 0;
}
