extern int __VERIFIER_nondet_int(void);
#include <assert.h>

int (*fp)(int);

int main(void)
{
  int x = 0;
  int ret = fp(x);
  while (__VERIFIER_nondet_int())
  {
    x = x - 1;
    assert(x >= 0);
  }
  return 0;
}
