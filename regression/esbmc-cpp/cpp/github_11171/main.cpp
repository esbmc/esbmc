// #11171: T() and new T() zero-initialise a class whose default constructor is
// trivial ([dcl.init]/8).
#include <cassert>

struct P
{
  int x;
};

struct D
{
  int x;
  int a[3];
  P p;
  D() = default;
};

struct E
{
};

int first(P p)
{
  return p.x;
}

int main()
{
  P p = P();
  assert(p.x == 0);

  P *h = new P();
  assert(h->x == 0);
  delete h;

  D d = D();
  assert(d.x == 0 && d.a[2] == 0 && d.p.x == 0);

  D *hd = new D();
  assert(hd->x == 0 && hd->a[1] == 0 && hd->p.x == 0);
  delete hd;

  assert(P().x == 0);
  assert(first(P()) == 0);

  E e = E();
  E *he = new E();
  delete he;
  return 0;
}
