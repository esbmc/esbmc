// Range-for over std::vector with initializer list
#include <vector>
#include <cassert>

int main()
{
  std::vector<int> v = {10, 20, 30, 40};
  int sum = 0;
  for (int x : v)
  {
    sum = sum + x;
  }
  assert(sum == 100);
  return 0;
}
