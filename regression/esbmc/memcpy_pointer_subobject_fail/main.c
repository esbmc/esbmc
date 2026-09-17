// The same copy under a truncating --unwind: without the structural graft the
// call falls back to __memcpy_impl's byte loop, whose truncation assumes false
// and vacuously proves the assertion below.
#include <assert.h>
#include <string.h>

struct S
{
  int *p;
};

int main()
{
  int x = -2;
  struct S s = {&x};
  int *q;
  memcpy(&q, &s, sizeof q);
  assert(*q == 5);
  return 0;
}
