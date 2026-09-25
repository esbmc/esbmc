// Placement new ([expr.new]/11): constructs at the given address with no
// allocation; the result aliases the placement argument. Covers primitive
// initializers, class constructors, the discarded-statement form, and the
// std::nothrow tag (which is NOT memory placement and keeps allocating).
#include <new>
#include <cassert>

struct S
{
  int v;
  S(int x) : v(x) {}
};

struct Self
{
  Self *self;
  Self() : self(this) {}
};

int main()
{
  int x = 0;
  int *p = new (&x) int(41);
  assert(x == 41);
  assert(p == &x);

  char buf[sizeof(S)];
  S *q = new ((void *)buf) S(7);
  assert(q->v == 7);
  assert((void *)q == (void *)buf);

  S s(1);
  new (&s) S(9);
  assert(s.v == 9);

  Self t;
  new (&t) Self;
  assert(t.self == &t); // `this` in the ctor is the placed object

  int *h = new (std::nothrow) int(5);
  if (h)
    assert(*h == 5);
  return 0;
}
