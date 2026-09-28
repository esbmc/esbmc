/* Neither assert(*p == 1) nor the checks raised inside it can fail. */
#include <assert.h>

int nondet_int();

int main()
{
  int x = 1, y = 1;
  int *p = nondet_int() ? &x : &y;
  assert(*p == 1);
  return 0;
}
