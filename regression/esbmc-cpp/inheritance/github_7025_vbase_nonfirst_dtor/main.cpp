// esbmc/esbmc#7025: the base constructor and the base destructor must be
// handed the same `this`. Taking one displacement from ESBMC's layout and the
// other from clang's ABI offset put ~B and B on different bytes.
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
  int flag;
  B()
  {
    flag = 1;
  }
  ~B()
  {
    assert(flag == 1);
  }
};
struct D : A, B
{
};

int main()
{
  D d;
  return 0;
}
