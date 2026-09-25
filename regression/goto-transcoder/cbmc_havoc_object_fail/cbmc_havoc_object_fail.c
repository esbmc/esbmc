#include <assert.h>

void __CPROVER_havoc_object(void *);

int main(void)
{
  int a[3] = {1, 2, 3};
  // The operand is &a[0] after the array decay, but havoc_object drops the
  // whole object, so a[2] is nondet here too.
  __CPROVER_havoc_object(a);
  assert(a[2] == 3);
  return 0;
}
