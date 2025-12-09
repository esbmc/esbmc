#include <assert.h>

int foo(int n)
{
  // n: [1,40]
  assert(n == 42); // always false
}


int main()
{
  foo(1);
  foo(40);
}
