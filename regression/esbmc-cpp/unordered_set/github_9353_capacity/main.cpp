// #9353: insert of a key already present in a full set is not an overflow.
#include <cassert>
#include <unordered_set>

int main()
{
  std::unordered_set<int> s;
  for (int i = 0; i < 64; ++i)
    s.insert(i);
  auto r = s.insert(5);
  assert(!r.second && *r.first == 5);
  assert(s.size() == 64);
  return 0;
}
