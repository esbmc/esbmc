// github #9406: erase(first, last) requires [first, last) to be a valid range.
#include <map>

int main()
{
  std::multimap<int, int> m;
  m.insert({1, 10});
  m.insert({2, 20});
  m.erase(m.end(), m.begin());
  return 0;
}
