// github #9406: erase(q) requires a dereferenceable q; end() is not one.
#include <map>

int main()
{
  std::multimap<int, int> m;
  m.insert({1, 10});
  m.insert({2, 20});
  m.erase(m.find(3));
  return 0;
}
