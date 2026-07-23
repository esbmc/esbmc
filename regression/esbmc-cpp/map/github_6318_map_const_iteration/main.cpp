// github #6318 (map part): const forward iteration over a std::map / std::multimap
// must compile. const_iterator was a typedef of the mutable iterator (non-const
// key/value pointers), and there were no const begin()/end() overloads. Add a
// distinct const_iterator (const element pointers, const-returning operator*/->)
// and const begin()/end(), so const iteration works and is truly read-only.
#include <cassert>
#include <map>

int sum_values(const std::map<int, int> &m)
{
  int t = 0;
  for (auto &kv : m) // range-for over a const map
    t += kv.second;
  return t;
}

int sum_keys(const std::map<int, int> &m)
{
  int s = 0;
  for (std::map<int, int>::const_iterator it = m.begin(); it != m.end(); ++it)
    s += it->first;
  return s;
}

int mm_count(const std::multimap<int, int> &m)
{
  int n = 0;
  for (auto &kv : m)
    n++;
  return n;
}

int main()
{
  std::map<int, int> m;
  m[1] = 10;
  m[2] = 20;
  assert(sum_values(m) == 30);
  assert(sum_keys(m) == 3);

  const std::map<int, int> &cm = m;
  assert(cm.find(1) != cm.end()); // const find/end
  assert(cm.find(9) == cm.end());

  std::multimap<int, int> mm;
  mm.insert({1, 10});
  mm.insert({1, 20});
  assert(mm_count(mm) == 2);
  return 0;
}
