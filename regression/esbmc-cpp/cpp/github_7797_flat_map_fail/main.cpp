// #7797: erasing a key must move the later mapped values with their keys.
#include <flat_map>
#include <cassert>

int main()
{
  std::flat_map<int, char> m;
  m[1] = 'a';
  m[2] = 'b';
  m[3] = 'c';
  m.erase(1);
  assert(m.values()[0] == 'a');
  return 0;
}
