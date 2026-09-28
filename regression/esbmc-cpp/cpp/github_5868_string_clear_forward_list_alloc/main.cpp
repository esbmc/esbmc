#include <string>
#include <forward_list>
#include <memory>
#include <cassert>

int main()
{
  std::string s("abc");
  assert(s.size() == 3);
  s.clear();
  assert(s.size() == 0);
  assert(s.empty());

  std::string t("hello");
  t.clear();
  assert(t.length() == 0);

  // [forward.list.overview]: forward_list takes an Allocator parameter.
  std::forward_list<int, std::allocator<int>> f;
  f.push_front(1);
  f.push_front(2);
  assert(f.front() == 2);

  std::forward_list<int> g;
  g.push_front(7);
  assert(g.front() == 7);
  return 0;
}
