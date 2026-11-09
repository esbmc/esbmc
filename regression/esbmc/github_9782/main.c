// #9782: only the path that runs j's initializer may rely on its value
#include <assert.h>
int nondet_int(void);
int main(void)
{
  int took = nondet_int();
  if (took)
    goto J;
  unsigned char j = 3;
J:
  if (!took)
    assert(j == 3);
  return 0;
}
