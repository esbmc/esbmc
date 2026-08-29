#include <vector>
#include <cassert>

int main()
{
  std::vector<double> d(3, 2);
  assert(d[1] == 3.0);
  return 0;
}
