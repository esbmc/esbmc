// github #9381: multimap::erase(k) erases every element with key k.
#include <cassert>
#include <map>

int main()
{
  std::multimap<int, int> m;
  m.insert({1, 10});
  m.insert({1, 11});
  m.insert({2, 20});
  m.erase(1);
  assert(m.count(1) != 0);
  return 0;
}
