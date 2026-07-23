// Negative counterpart: after `throw PA()` caught as A& (A at a non-zero offset
// in PA), a.a is 1, never P's p (9). Asserting the P value must FAIL, proving
// the binding re-bases to the A subobject rather than aliasing P's memory.
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

struct PA : P, A
{
};

int main()
{
  try
  {
    throw PA();
  }
  catch (A &a)
  {
    assert(a.a == 9);
  }
  return 0;
}
