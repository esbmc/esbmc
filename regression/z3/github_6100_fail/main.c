#include <stdbool.h>

bool eq(int a, int b)
{
  return a == b;
}

int main()
{
  int var;
  __ESBMC_assert(__ESBMC_forall(&var, eq(var, 6)), "forall-eq-6");
}
