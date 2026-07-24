#include <vector>
#include <cassert>

int main()
{
  std::vector<int> v;
  v.emplace_back(5);
  assert(v[0] == 99); // wrong value
  return 0;
}
