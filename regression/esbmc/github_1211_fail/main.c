#include <assert.h>

int nondet_int(void);

int main(void)
{
  int x = nondet_int();
  int y = nondet_int();
  assert(x != 5);
  if (y > 0)
    assert(x != 5);
  return 0;
}
