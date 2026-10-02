// new T[n][m] of a class constructs and destroys all n * m elements.
#include <cassert>
int live = 0;
struct C
{
  int v;
  C() : v(9) { ++live; }
  ~C() { --live; }
};
struct S
{
  int a = 5;
  int b;
};
int main()
{
  C (*p)[2] = new C[3][2];
  assert(live == 6 && p[2][1].v == 9);
  delete[] p;
  assert(live == 0);
  S (*q)[3] = new S[2][3]();
  assert(q[1][2].a == 5);
  delete[] q;
  return 0;
}
