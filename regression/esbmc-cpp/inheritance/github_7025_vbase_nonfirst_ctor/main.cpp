// esbmc/esbmc#7025: the same displacement applies to the base constructor
// `this` and to pointer and reference upcasts, not only to member calls.
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
  B()
  {
    buf[0] = 'z';
  }
  void put(char c)
  {
    buf[0] = c;
  }
  char get() const
  {
    return buf[0];
  }
};
struct D : A, B
{
};

int main()
{
  D d;
  assert(d.buf[0] == 'z');
  d.put('A');
  assert(d.get() == 'A');

  B *pb = &d;
  pb->put('Q');
  assert(d.buf[0] == 'Q');

  B &rb = d;
  rb.put('R');
  assert(d.buf[0] == 'R');
  return 0;
}
