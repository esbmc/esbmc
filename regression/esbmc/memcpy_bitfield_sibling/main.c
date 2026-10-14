// A bitfield reports a byte range it does not own, so grafting one would drop
// the siblings sharing its bytes. Both members must survive the copy.
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
  assert(x.a == 5);
  assert(x.b == 7);
  return 0;
}
