#include <cassert>

int dtors = 0, g = 10;

struct C
{
  int v;
  C(int x) : v(x) {}
  ~C()
  {
    dtors++;
    g = 0;
  }
};

struct L
{
  ~L() { dtors += 100; }
};

int ref(const C &c) { return c.v; }
int member() { return C(1).v; }
int bound() { return ref(C(2)); }
int before_locals()
{
  L l;
  return g + C(3).v;
}

struct P
{
  int *p;
  P(int x) : p(new int(x)) {}
  ~P() { delete p; }
};

int conditional(bool c) { return c ? *P(6).p : 0; }

int main()
{
  assert(member() == 1);
  assert(dtors == 1);
  assert(bound() == 2);
  assert(dtors == 2);
  g = 10;
  assert(before_locals() == 13);
  assert(dtors == 103);
  assert(conditional(false) == 0);
}
