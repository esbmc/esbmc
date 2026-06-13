// dynamic_cast across multiple inheritance with non-zero base offsets:
// a down-cast from a non-first base back to the derived type, and a sibling
// cross-cast between two bases. Both require the result pointer to be re-offset
// to the correct sub-object inside the runtime type (result = src - off(S) +
// off(T)), not a plain reinterpret.
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
  int z;
  D() : z(3)
  {
  }
};

int main()
{
  D d;
  B2 *b2 = &d; // upcast to the non-first base (non-zero offset)

  D *dd = dynamic_cast<D *>(b2); // down-cast through the offset
  assert(dd && dd->x == 1 && dd->y == 2 && dd->z == 3);

  B1 *b1 = dynamic_cast<B1 *>(b2); // sibling cross-cast B2* -> B1*
  assert(b1 && b1->x == 1);

  return 0;
}
