#include <cassert>

struct a
{
  a(int, int *c) : b(*c)
  {
  }
  int b;
};
int f = 42;
int main()
{
  a a(0, &f);
  assert(a.b == 42);
}
