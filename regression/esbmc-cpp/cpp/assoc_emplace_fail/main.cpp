#include <map>
#include <cassert>

int main()
{
  std::map<int, int> m;
  m.emplace(1, 2);
  assert(m[1] == 99); // wrong value
  return 0;
}
