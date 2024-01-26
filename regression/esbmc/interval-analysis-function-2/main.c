#include <assert.h>
int foo() {
  return 42;
}

int main()
{
  int A = foo();
  assert(A == 42);
  A = foo();
  assert(A > 40);
}


