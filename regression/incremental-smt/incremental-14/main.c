#include <assert.h>

int foo(int x)
{
  int res = 0;
  for(int i = 0; i < x; i++)
  {
    res += i;
    assert(res >= 0);
  }
  return res;
}

int main()
{
  int x = nondet_int();
  __VERIFIER_assume(x > 0 && x < 100);
  if(x < 0)
    assert(foo(100000) > 0);
  else
    assert(x > 0);
  return 0;
}
