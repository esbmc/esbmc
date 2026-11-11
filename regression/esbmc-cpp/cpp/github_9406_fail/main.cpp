// github #9406: erasing the second of two equal keys keeps the first.
#include <cassert>
#include <map>

int main()
{
  std::multimap<int, int> m;
  m.insert({1, 10});
  m.insert({1, 11});
  auto it = m.begin();
  ++it;
  m.erase(it);
  assert(m.begin()->second == 11);
  return 0;
}
