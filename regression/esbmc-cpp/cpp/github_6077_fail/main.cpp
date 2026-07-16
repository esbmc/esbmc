#include <assert.h>
int dtor_count;

class A
{
public:
  int num;
  A(int n) : num(n) {}
  ~A() { dtor_count++; }
};

int check()
{
  A a(1);
  return 0;
}

int main()
{
  check();
  assert(dtor_count == 0); // must fail: check()'s local was destructed
  return 0;
}
