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
  assert(dtor_count == 1);
  func(2);
  func(3);
  assert(dtor_count == 3);
  A(7);
  assert(dtor_count == 4);
  (void)func(5);
  assert(dtor_count == 5);
  dtor_count > 0 ? func(6) : func(7);
  assert(dtor_count == 6);
  return 0;
}
