#include <assert.h>

int foo(int n)
{
  // n: [51,60]
  assert(n == 42); // always holds
}


int main()
{
  foo(51);
  foo(60);
}
