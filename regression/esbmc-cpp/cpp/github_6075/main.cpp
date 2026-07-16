#include <assert.h>
int dtor_count;

class A
{
public:
  int num;
  A(int n) : num(n) {}
  ~A() { dtor_count++; }
};

A func(int n)
{
  return A(n);
}

int main()
{
  int r = func(5).num + func(6).num;
  assert(r == 11);
  assert(dtor_count == 2); // both temporaries died at the end of the decl

  const A &ref = func(7); // lifetime extension: no destruction here
  assert(ref.num == 7);
  assert(dtor_count == 2);
  return 0;
}
