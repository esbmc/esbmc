#include <assert.h>

int foo(int n)
{
  // n: [42,42]
  assert(n == 42); // always holds
}


int main()
{
  foo(42);
}
