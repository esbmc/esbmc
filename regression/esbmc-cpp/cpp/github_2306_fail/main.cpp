#include <cassert>

int count;

class A
{
public:
  int num;
  A(int n) : num(n)
  {
  }
  ~A()
  {
    count++;
  }
};

A func(int n)
{
  return A(n);
}

void dtor()
{
  A a = func(1);
  assert(a.num == 1);
}

int main()
{
  dtor();
  assert(count == 2); // deliberately wrong: count is actually 1
}
