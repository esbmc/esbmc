#include <deque>
#include <map>
#include <vector>
#include <cassert>

int main()
{
  const std::vector<int> v(3, 1), w(3, 1);
  assert(v == w);
  assert(!(v != w));

  const std::deque<int> p, q;
  assert(p == q);
  assert(!(p != q));

  std::multimap<int, int> a, b;
  a.insert(std::pair<int, int>(1, 2));
  b.insert(std::pair<int, int>(1, 2));
  const std::multimap<int, int> &ca = a;
  const std::multimap<int, int> &cb = b;
  assert(ca == cb);
  assert(!(ca != cb));
  return 0;
}
