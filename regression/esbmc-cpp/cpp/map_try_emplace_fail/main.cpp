#include <map>
#include <cassert>

int main()
{
  std::map<int, int> m;
  m.insert_or_assign(1, 9);
  assert(m[1] == 2); // wrong value
  return 0;
}
