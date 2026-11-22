#include <assert.h>
int nondet_int(void);
int main(void)
{
  int x = nondet_int();
  assert(x != 42);
  return 0;
}
