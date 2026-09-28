#include <cassert>
extern "C" int helper(int);
extern "C" int __VERIFIER_nondet_int(void);

int main()
{
  int x = __VERIFIER_nondet_int();
  assert(helper(x) != 42);
  return 0;
}
