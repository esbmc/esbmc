// #9378: erasing [begin + 1, begin + 3) from {0..4} keeps key 3.
#include <cassert>
#include <map>

int main()
{
  std::map<int, int> m;
  for (int i = 0; i < 5; i++)
    m[i] = i;
  std::map<int, int>::iterator f = m.begin();
  ++f;
  std::map<int, int>::iterator l = f;
  ++l;
  ++l;
  m.erase(f, l);
  assert(m.count(3) == 0);
  return 0;
}
