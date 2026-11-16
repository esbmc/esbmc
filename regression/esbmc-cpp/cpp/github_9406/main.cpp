// github #9406: multimap::erase(iterator) erases the element the iterator
// points to, not the first one with an equal key; erase(first, last) erases
// exactly [first, last) ([associative.reqmts] Table 91).
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
  assert(m.size() == 1 && m.begin()->second == 10);

  std::multimap<int, int> r;
  for (int i = 0; i < 5; i++)
    r.insert({i, i * 10});
  auto f = r.begin();
  ++f;
  auto l = f;
  ++l;
  ++l;
  r.erase(f, l);
  assert(r.size() == 3);
  auto p = r.begin();
  assert(p->first == 0);
  ++p;
  assert(p->first == 3 && p->second == 30);
  ++p;
  assert(p->first == 4);
  return 0;
}
