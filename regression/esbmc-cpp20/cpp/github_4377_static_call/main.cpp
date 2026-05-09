#include <cassert>

struct F
{
  static int operator()(int x) { return x + 1; }
};

struct A
{
  static int operator[](int i) { return i * 10; }
};

int main()
{
  F f;
  assert(f(2) == 3);

  A a;
  assert(a[3] == 30);

  return 0;
}
