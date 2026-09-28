#include <cassert>

int ctor_count;
int dtor_count;

class A
{
public:
  int num;
  A(int n) : num(n)
  {
    ctor_count++;
  }
  ~A()
  {
    dtor_count++;
  }
};

int helper(int x)
{
  return x + 1;
}

A func(int n)
{
  return A(helper(n));
}

void two_calls()
{
  A a = func(1);
  A b = func(2);
  assert(a.num == 2);
  assert(b.num == 3);
}

int main()
{
  two_calls();
  assert(ctor_count == 2);
  assert(dtor_count == 2);
  return 0;
}
