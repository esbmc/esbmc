#include <list>
#include <vector>
#include <iterator>
#include <cassert>

int main()
{
  // [iterator.operations]: a bidirectional iterator has no operator+, so
  // advance has to step with ++ and --.
  std::list<int> l;
  l.push_back(1);
  l.push_back(2);
  l.push_back(3);

  std::list<int>::iterator it = l.begin();
  std::advance(it, 2);
  assert(*it == 3);
  std::advance(it, -1);
  assert(*it == 2);
  std::advance(it, 0);
  assert(*it == 2);

  assert(*std::next(l.begin()) == 2);
  assert(*std::prev(l.end()) == 3);

  // Random access keeps the i + n path.
  std::vector<int> v;
  v.push_back(4);
  v.push_back(5);
  v.push_back(6);
  std::vector<int>::iterator vi = v.begin();
  std::advance(vi, 2);
  assert(*vi == 6);
  assert(*std::next(v.begin()) == 5);
  return 0;
}
