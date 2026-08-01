// Further shapes of the same binding, to keep the address-of distribution from
// being narrowed later: a nested conditional (the distribution has to recurse),
// arms that are struct members or array elements, and a scalar conditional,
// which shows the defect was never specific to class types.
#include <cassert>

int nondet_int();

struct P
{
  int v;
};

struct H
{
  P p, q;
};

void bump(P &x)
{
  x.v += 10;
}

void bump_int(int &x)
{
  x += 10;
}

int main()
{
  P a{1}, b{2}, c{3};
  int i = 2;
  bump(i < 1 ? a : (i < 3 ? b : c));
  assert(a.v == 1 && b.v == 12 && c.v == 3);

  H h;
  h.p.v = 1;
  h.q.v = 2;
  bump(i < 1 ? h.p : h.q);
  assert(h.p.v == 1 && h.q.v == 12);

  P arr[2];
  arr[0].v = 1;
  arr[1].v = 2;
  int n = nondet_int();
  bump(n < 1 ? arr[0] : arr[1]);
  assert((arr[0].v == 11 && arr[1].v == 2) || (arr[0].v == 1 && arr[1].v == 12));

  int s = 1, t = 2;
  bump_int(i < 1 ? s : t);
  assert(s == 1 && t == 12);

  return 0;
}
