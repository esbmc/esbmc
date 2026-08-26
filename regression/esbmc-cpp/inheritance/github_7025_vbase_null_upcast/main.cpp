// esbmc/esbmc#7025: [conv.ptr]/3 - a null derived pointer upcasts to a null
// base pointer, so the subobject displacement must not be applied to it.
#include <cassert>

struct V
{
  int v;
};
struct A
{
  int a;
};
struct B : virtual V
{
  char buf[8];
};
struct D : A, B
{
};

int main()
{
  D *pd = 0;
  B *pb = pd;
  assert(pb == 0);
  return 0;
}
