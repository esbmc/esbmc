#include <assert.h>
int foo() {
  return 22;
}

int main()
{
  int A = foo();
  assert(A == 42);
  A = foo();
  assert(A > 40);
}


