#include <list>
#include <deque>
#include <map>
#include <set>
#include <vector>
#include <cassert>

int main()
{
  std::list<int> l;
  l.push_back(7);
  const std::list<int>::iterator li = l.begin();
  assert(*li == 7);

  std::deque<int> d;
  d.push_back(8);
  const std::deque<int>::iterator di = d.begin();
  assert(*di == 8);

  std::vector<int> v;
  v.push_back(9);
  const std::vector<int>::iterator vi = v.begin();
  assert(*vi == 9);

  std::set<int> s;
  s.insert(10);
  const std::set<int>::iterator si = s.begin();
  assert(*si == 10);

  std::map<int, int> m;
  m[1] = 11;
  const std::map<int, int>::iterator mi = m.begin();
  assert(mi->second == 11);
  assert((*mi).first == 1);
  return 0;
}
