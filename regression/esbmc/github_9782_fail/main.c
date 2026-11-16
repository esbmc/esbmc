// #9782: the goto skips j's initializer, so j is indeterminate at J
#include <assert.h>
int nondet_int(void);
int main(void)
{
  if (nondet_int())
    goto J;
  unsigned char j = 3;
J:
  assert(j >= 3);
  return 0;
}
