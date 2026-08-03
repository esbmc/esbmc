#include <assert.h>

// A read through p is resolved once and memoised. Retargeting p must drop
// that entry, or the second read answers with the first object's value.
int a = 1, b = 2;

int main()
{
  int *p = &a;
  int first = *p;
  p = &b;
  int second = *p;

  assert(first == 1);
  assert(second == 2);
  return 0;
}
