#include <assert.h>
extern int helper(int);
int __VERIFIER_nondet_int(void);

int main(void)
{
  int x = __VERIFIER_nondet_int();
  assert(helper(x) != 42);
  return 0;
}
