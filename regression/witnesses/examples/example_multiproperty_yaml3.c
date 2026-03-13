#include <assert.h>
#include <stdlib.h>

int nondet_int(void);
void __VERIFIER_error(void) { assert(0); }

int main()
{
  int a = nondet_int(), b = nondet_int();
  int result = a + b;
  int arr[3] = {0, 1, 2};
  int *p = 0;
  int z;

  if (a > 10)
    arr[3] = result;

  if (a == -b)
    z = 100 / (result);

  if (b < 0)
    *p = result;

  if (a > b)
    result = z + 1;

  if (a == 1 && b == 2)
    __VERIFIER_error();

  return result + 2147483647;
}