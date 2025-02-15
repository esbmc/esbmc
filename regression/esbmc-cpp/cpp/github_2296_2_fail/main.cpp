#include <cassert>

template <typename a>
struct b
{
  a c;
  int d;
};
struct d;
struct e
{
  void f(d);
};
struct g
{
  e h;
};
struct d : g
{
};
b<g> k;

int main()
{
  k.d = 0;
  assert(k.d != 0); // fail
}
