#include <assert.h>

int nondet_int(void);
void __VERIFIER_error(void) { assert(0); }

int main()
{
  int a = nondet_int(), b = nondet_int();
  int result = a + b;

  int arr[2] = {0, 1};
  int *p = 0;
  int x;

  int y = result + 2147483647;

  if (result > 0)
    arr[2] = y;

  if (a == -b)
    x = 10 / (a + b);

  if (b > 100)
    *p = 1;

  if (a < 0)
    result = x + 1;

  if (a == 0 && b == 0)
    __VERIFIER_error();

  return result;
}