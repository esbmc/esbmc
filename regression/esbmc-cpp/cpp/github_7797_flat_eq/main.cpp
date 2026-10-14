// #7797: flat_set equality compares elements, not comparator equivalence, and
// a const container reaches its lookups. [tab:container.req], [flat.set.defn].
#include <flat_set>
#include <flat_map>
#include <cassert>

// Orders on the last decimal digit, so 3 and 13 are equivalent but not equal.
struct mod10
{
  bool operator()(int a, int b) const
  {
    return a % 10 < b % 10;
  }
};

int main()
{
  std::flat_set<int, mod10> a, b;
  a.insert(3);
  b.insert(13);
  assert(a.size() == 1 && b.size() == 1);
  // Equivalent under the comparator, so each rejects the other as a duplicate.
  assert(!a.insert(13).second);
  assert(a != b);

  std::flat_set<int, mod10> c;
  c.insert(3);
  assert(a == c);

  const std::flat_set<int, mod10> &ca = a;
  assert(ca.contains(3) && ca.count(3) == 1);
  assert(*ca.begin() == 3);
  assert(*ca.find(13) == 3);

  std::flat_map<int, char> m;
  m[1] = 'a';
  m[2] = 'b';
  const std::flat_map<int, char> &cm = m;
  assert(cm.find(2)->second == 'b');
  assert(cm.lower_bound(1)->first == 1);
  assert(cm.equal_range(2).first->second == 'b');
  assert(cm.at(1) == 'a');

  // at() on an absent key throws, as [flat.map.access] requires.
  bool caught = false;
  try
  {
    m.at(9);
  }
  catch (std::out_of_range &)
  {
    caught = true;
  }
  assert(caught);
  return 0;
}
