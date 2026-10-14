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
  assert(dtor_count == 0); // must fail: temporaries destructed at end of decl
  return 0;
}
