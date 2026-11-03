// #9353: on a full map, operator[] returned one shared static value, so
// writes to two new keys aliased and this assertion was proved.
#include <cassert>
#include <unordered_map>

int main()
{
  std::unordered_map<int, int> m;
  for (int i = 0; i < 64; ++i)
    m[i] = i;
  m[100] = 1;
  m[200] = 2;
  assert(m[100] == 2);
  return 0;
}
