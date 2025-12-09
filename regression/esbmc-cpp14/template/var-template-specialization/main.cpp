#include <assert.h>

template <typename>
int a;

int main()
{
  a<int> = 2;
  a<double> = 42;
  assert(a<int> == 2);
  assert(a<int> != 42);
  assert(a<double> == 42);
  assert(a<double> != 2);
}