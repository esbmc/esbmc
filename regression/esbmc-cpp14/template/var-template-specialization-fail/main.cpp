#include <assert.h>

template <typename>
int a;

int main()
{
  a<int> = 2;
  a<double> = 3;
  assert(a<int> == 3);
  assert(a<double> == 4);
}