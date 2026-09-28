#include <assert.h>
extern int clamp(int);
int __VERIFIER_nondet_int(void);

int main(void)
{
  int x = __VERIFIER_nondet_int();
  int y = clamp(x);
  assert(y >= 0 && y <= 10);
  return 0;
}
