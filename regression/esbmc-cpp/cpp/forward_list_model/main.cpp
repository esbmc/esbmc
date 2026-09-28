// Model for <forward_list>, which was missing entirely (issue #5868).
//
// A singly-linked list over a fixed pool with index links rather than
// pointers, so the structure survives a copy -- a self-referential pointer
// would dangle into the source object. std::forward_list deliberately has no
// size(), and expresses mutation relative to the *preceding* position, which
// is what before_begin and the _after operations are for.
#include <forward_list>

int main()
{
  std::forward_list<int> l;
  __ESBMC_assert(l.empty(), "a default-constructed list is empty");

  l.push_front(3);
  l.push_front(2);
  l.push_front(1);
  __ESBMC_assert(!l.empty(), "no longer empty");
  __ESBMC_assert(l.front() == 1, "push_front puts the newest at the front");

  // Traversal reaches every element, in push order reversed.
  int seen[3];
  int n = 0;
  for (std::forward_list<int>::iterator it = l.begin(); it != l.end(); ++it)
    seen[n++] = *it;
  __ESBMC_assert(n == 3, "iteration visits every element exactly once");
  __ESBMC_assert(
    seen[0] == 1 && seen[1] == 2 && seen[2] == 3, "and in the right order");

  l.pop_front();
  __ESBMC_assert(l.front() == 2, "pop_front removes the front");

  // insert_after places the element *after* the given position, and
  // before_begin() addresses the position preceding the first element.
  std::forward_list<int> m;
  m.push_front(3);
  m.push_front(1);
  m.insert_after(m.begin(), 2);
  std::forward_list<int>::iterator second = m.begin();
  ++second;
  __ESBMC_assert(m.front() == 1, "insert_after leaves the head alone");
  __ESBMC_assert(*second == 2, "and places the element after it");

  m.insert_after(m.before_begin(), 0);
  __ESBMC_assert(m.front() == 0, "before_begin inserts at the head");

  // erase_after removes the element following the position, and returns the
  // one after that.
  std::forward_list<int>::iterator after = m.erase_after(m.begin());
  __ESBMC_assert(m.front() == 0, "erase_after leaves the position itself");
  __ESBMC_assert(*after == 2, "and returns the element that follows");

  // A copy is independent of its source: the index links must not alias back
  // into the original's pool.
  std::forward_list<int> a;
  a.push_front(1);
  std::forward_list<int> b = a;
  b.push_front(2);
  __ESBMC_assert(a.front() == 1, "the source is unchanged by a copy's push");
  __ESBMC_assert(b.front() == 2, "and the copy has its own front");

  std::forward_list<int> c;
  c = a;
  c.push_front(9);
  __ESBMC_assert(a.front() == 1, "assignment is independent too");

  l.clear();
  __ESBMC_assert(l.empty(), "clear empties the list");
  return 0;
}
