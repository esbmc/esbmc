//test_abort.c:
#include "test_abort.h"

void func1(int y);
void func1(int y)
{
  __ESBMC_assume(y >= 0);
  __ESBMC_assume(y < 1);
  func2(y);
}
