// x.a really is overwritten by the copy, so proving it still zero would mean a
// grafted bitfield had silently dropped its byte-sharing sibling.
#include <assert.h>
#include <string.h>

struct A
{
  unsigned a : 3;
  unsigned b : 29;
};

struct B
{
  unsigned c : 3;
  unsigned d : 29;
};

int main()
{
  struct A x = {0, 0};
  struct B y = {5, 7};
  memcpy(&x, &y, sizeof x);
  assert(x.a == 0);
  return 0;
}
