// Negative companion: after the sibling cross-cast B2* -> B1*, b1 points at the
// B1 sub-object, so b1->x is 1, not 2. Asserting the wrong value must fail
// (proves the cross-cast re-offsets rather than aliasing B2's data).
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
  assert(b1 && b1->x == 2); // wrong: b1->x is 1
  return 0;
}
