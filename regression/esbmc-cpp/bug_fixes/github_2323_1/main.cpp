#include <cassert>

template <typename>
struct b;
template <typename c>
struct a
{
  b<c> *d;
};
template struct a<int>;
template <typename c>
struct b : a<c>
{
};
b<int> x;

int main()
{
  x.d = 0;
  assert(x.d == 0);
}
