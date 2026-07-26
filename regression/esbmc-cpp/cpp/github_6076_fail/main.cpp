#include <assert.h>
int dtor_count;

class A
{
public:
  int num;
  A(int n) : num(n) {}
  ~A() { dtor_count++; }
};

A func(int n) { return A(n); }

int main()
{
  func(1);
  assert(dtor_count == 0); // must fail: the discarded temporary is destructed
  return 0;
}
