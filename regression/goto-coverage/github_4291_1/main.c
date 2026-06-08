#include <stdbool.h>

void __VERIFIER_assume(int cond);
int __VERIFIER_nondet_int(void);

bool foo(int x)
{
  if (x > 0 && x < 100)
  {
    return true;
  }
  return false;
}

int main(void)
{
  int x = __VERIFIER_nondet_int();
  __VERIFIER_assume(x >= -1000);
  __VERIFIER_assume(x <= 1000);
  foo(x);
  return 0;
}
