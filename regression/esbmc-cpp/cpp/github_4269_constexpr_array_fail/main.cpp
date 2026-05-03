#include <array>

int main()
{
  std::array<int, 3> a = {1, 2, 3};
  unsigned i;
  __ESBMC_assume(i < 10);
  int x = a[i];
  return x;
}
