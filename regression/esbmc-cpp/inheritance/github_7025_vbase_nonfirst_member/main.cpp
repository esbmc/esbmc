// esbmc/esbmc#7025: a class with a virtual base resolves its own members to the
// wrong offset when it is a non-first base and the access goes through `this`.
// The write in B::put and the read through `d` must land on the same bytes.
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
  void put(char c)
  {
    buf[0] = c;
  }
};
struct D : A, B
{
};

int main()
{
  D d;
  d.buf[0] = 'x';
  d.put('A');
  assert(d.buf[0] == 'A');
  return 0;
}
