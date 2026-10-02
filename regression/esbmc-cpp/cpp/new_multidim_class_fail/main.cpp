// new T[n][m] of a class constructs and destroys all n * m elements.
#include <cassert>
int live = 0;
struct C
{
  int v;
  C() : v(9) { ++live; }
  ~C() { --live; }
};
int main()
{
  C (*p)[2] = new C[3][2];
  delete[] p;
  assert(live != 0);
  return 0;
}
