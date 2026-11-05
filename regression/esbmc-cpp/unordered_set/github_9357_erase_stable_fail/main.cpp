// #9357: an iterator to an element survives the erase of another one.
#include <cassert>
#include <unordered_set>

int main()
{
  std::unordered_set<int> s;
  s.insert(1);
  s.insert(2);
  s.insert(3);
  auto it = s.find(2);
  s.erase(1);
  assert(*it != 2);
  return 0;
}
