#include <cassert>

struct S
{
  int a;
  int b;
};

struct U
{
  int a;
  U() = default;
};

struct T
{
  int a;
  T()
  {
    a = 7;
  }
};

int main()
{
  S *s = new S[2]();
  assert(s[0].a == 0 && s[0].b == 0 && s[1].a == 0 && s[1].b == 0);
  delete[] s;

  U *u = new U[2]();
  assert(u[0].a == 0 && u[1].a == 0);
  delete[] u;

  T *t = new T[3]();
  assert(t[0].a == 7 && t[2].a == 7);
  delete[] t;

  return 0;
}
