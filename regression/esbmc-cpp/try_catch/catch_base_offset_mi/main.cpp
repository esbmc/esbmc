// Catching a multiple-inheritance thrown object by a base that sits at a
// non-zero offset inside the dynamic type must re-base the bound pointer to the
// correct subobject. `throw PA()` stores a pointer to the most-derived object
// (= the P subobject at offset 0); catching A& must yield value + off(A in PA),
// not a plain reinterpret. Both the offset-0 (first base) and offset-N (second
// base) directions are exercised. Regression for the catch-by-base offset bug.
#include <cassert>

struct A
{
  int a;
  A() : a(1)
  {
  }
};

struct P
{
  virtual ~P()
  {
  }
  int p;
  P() : p(9)
  {
  }
};

struct AP : A, P   // A first  -> off(A in AP) == 0
{
};

struct PA : P, A   // A second -> off(A in PA) != 0
{
};

int main()
{
  try
  {
    throw AP();
  }
  catch (A &a)
  {
    assert(a.a == 1);
  }

  try
  {
    throw PA();
  }
  catch (A &a)
  {
    assert(a.a == 1);
  }
  return 0;
}
