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

struct AP : A, P
{
};

int main()
{
  AP ap;
  A &as = ap;
  assert(as.a == 9);
}
