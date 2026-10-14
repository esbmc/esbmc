// github #6318 (set/multiset part): const iteration over a std::set / std::multiset
// must compile. The const begin/end/find/rbegin/rend overloads were declared but
// did not compile because const_iterator/const_reverse_iterator held a non-const
// element pointer. They now hold `const key_type*` and return const references,
// so const iteration works and is truly read-only (mutation is rejected).
#include <cassert>
#include <set>

int sum(const std::set<int> &s)
{
  int t = 0;
  for (auto &x : s) // range-for over a const set
    t += x;
  return t;
}

bool has(const std::set<int> &s, int k)
{
  return s.find(k) != s.end(); // const find/end
}

int msize(const std::multiset<int> &s)
{
  int n = 0;
  for (std::multiset<int>::const_iterator it = s.begin(); it != s.end(); ++it)
    n++;
  return n;
}

int main()
{
  std::set<int> s;
  s.insert(1);
  s.insert(2);
  s.insert(3);
  assert(sum(s) == 6);
  assert(has(s, 2));
  assert(!has(s, 9));
  assert(*s.begin() == 1); // via non-const, still fine

  const std::set<int> &cs = s;
  assert(*cs.begin() == 1); // const begin deref

  std::multiset<int> ms;
  ms.insert(5);
  ms.insert(5);
  assert(msize(ms) == 2);
  return 0;
}
