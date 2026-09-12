// A sibling cross-cast carries both base markers on one typecast
// (clang_cpp_convert_vft.cpp). Resolving only one leaves b1 on B2's storage.
#include <cassert>

struct B1
{
  virtual ~B1()
  {
  }
  int x;
  B1() : x(1)
  {
  }
};

struct B2
{
  virtual ~B2()
  {
  }
  int y;
  B2() : y(2)
  {
  }
};

struct D : B1, B2
{
};

int main()
{
  D d;
  B2 *b2 = &d;
  B1 *b1 = dynamic_cast<B1 *>(b2);
  assert(b1 && b1->x == 1);
}
