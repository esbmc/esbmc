#include <cassert>
#include <deque>
#include <list>
#include <set>
#include <forward_list>
int main() {
  std::deque<int> d{1,2,3};
  assert(d.size()==3 && d[0]==1 && d[2]==3);
  std::list<int> l{1,2,3};
  assert(l.size()==3 && l.front()==1 && l.back()==3);
  std::set<int> s{3,1,2};
  assert(s.size()==3 && *s.begin()==1);
  std::multiset<int> m{2,1,2};
  assert(m.size()==3 && *m.begin()==1);
  std::forward_list<int> f{1,2,3};
  assert(f.front()==1);
  return 0;
}
