// new T[n][m]() value-initialises every element of every row: members no
// default member initializer writes are zero ([dcl.init]/9).
#include <cassert>
struct S
{
  int a = 1;
  int b;
  int *q;
};
int main()
{
  S (*p)[3] = new S[2][3]();
  assert(p[1][2].a == 1 && p[1][2].b == 0 && p[0][1].q == nullptr);
  delete[] p;
  return 0;
}
