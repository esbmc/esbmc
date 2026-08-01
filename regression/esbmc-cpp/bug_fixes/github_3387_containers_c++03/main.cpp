// Pin issue #3387: the C++98 container headers must parse and verify under
// --std c++03.  <vector> used nullptr and an unguarded variadic emplace_back,
// <map> an unguarded initializer_list ctor, <stack>/<queue> `>>` where C++03
// needs `> >`, and <typeinfo> spelled its what() overrides with a macro that
// vanished in C++03 and so relaxed the base's throw() specification.
// <list>, <map>, <queue> and <stack> all include <vector>, so they failed too.
#include <cassert>
#include <vector>
#include <list>
#include <map>
#include <set>
#include <deque>
#include <stack>
#include <queue>
#include <typeinfo>

int main()
{
  std::vector<int> v;
  v.push_back(1);
  v.push_back(2);
  assert(v.size() == 2 && v[0] == 1 && v[1] == 2);

  std::list<int> l;
  l.push_back(7);
  assert(l.size() == 1 && l.front() == 7);

  std::map<int, int> m;
  m[3] = 30;
  assert(m.size() == 1 && m[3] == 30);

  std::set<int> s;
  s.insert(5);
  assert(s.size() == 1);

  std::deque<int> d;
  d.push_back(9);
  assert(d.size() == 1 && d[0] == 9);

  std::stack<int> st;
  st.push(4);
  assert(st.size() == 1 && st.top() == 4);

  std::queue<int> q;
  q.push(6);
  assert(q.size() == 1 && q.front() == 6);

  const std::type_info &ti = typeid(int);
  assert(ti == typeid(int));

  return 0;
}
