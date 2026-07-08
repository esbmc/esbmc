/* Null restrict pointers designate no object and cannot violate the contract;
   the non-null guard must keep this from being flagged. */
#include <stddef.h>

void f(int *restrict a, int *restrict b)
{
  if (a && b)
  {
    *a = 1;
    *b = 2;
  }
}

int main(void)
{
  f(NULL, NULL);
  return 0;
}
