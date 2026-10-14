#include <deque>
#include <set>
#include <map>
#include <cassert>

int main()
{
  std::deque<int> d;
  d.push_back(1);
  d.push_back(2);
  int ds = 0;
  for (std::deque<int>::const_iterator it = d.cbegin(); it != d.cend(); ++it)
    ds += *it;
  assert(ds == 3);

  std::set<int> s;
  s.insert(1);
  s.insert(2);
  int ss = 0;
  for (std::set<int>::const_iterator it = s.cbegin(); it != s.cend(); ++it)
    ss += *it;
  assert(ss == 3);
  assert(*s.crbegin() == 2);

  std::multiset<int> ms;
  ms.insert(5);
  ms.insert(5);
  int mc = 0;
  for (std::multiset<int>::const_iterator it = ms.cbegin(); it != ms.cend();
       ++it)
    mc++;
  assert(mc == 2);

  std::map<int, int> m;
  m[1] = 10;
  m[2] = 20;
  int msum = 0;
  for (std::map<int, int>::const_iterator it = m.cbegin(); it != m.cend(); ++it)
    msum += it->second;
  assert(msum == 30);
  assert(m.crbegin()->second == 20);

  std::multimap<int, int> mm;
  mm.insert(std::make_pair(1, 2));
  assert(mm.cbegin()->second == 2);

  return 0;
}
