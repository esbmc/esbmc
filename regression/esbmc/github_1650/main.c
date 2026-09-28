#include <assert.h>
extern void __VERIFIER_assume(int cond);
#include <stdlib.h>
int main() {
  int *ptr = malloc(sizeof(int));
  *ptr = 0;
  int a[1] = {5};
  a[*ptr] = 5;
  __VERIFIER_assume(a[*ptr] < 5 || a[*ptr] < 2147483647 || a[*ptr] > 1 || a[*ptr] >= 1);
  assert(a[*ptr] > 2147483646);
  if (a[*ptr]) {}
  return 0;
}
