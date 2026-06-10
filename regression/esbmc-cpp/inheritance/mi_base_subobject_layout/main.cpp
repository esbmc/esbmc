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

struct PA : P, A
{
};

int main()
{
  AP ap;
  A &ap_a = ap;
  P &ap_p = ap;
  assert(ap_a.a == 1);
  assert(ap_p.p == 9);

  PA pa;
  A &pa_a = pa;
  P &pa_p = pa;
  assert(pa_a.a == 1);
  assert(pa_p.p == 9);
}
