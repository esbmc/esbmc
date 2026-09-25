#include <unordered_set>
#include <cassert>

int main()
{
  // The container holds at least the 64 slots its siblings do.
  std::unordered_set<int> s;
  for (int i = 0; i < 64; i++)
    s.insert(i);
  assert(s.size() == 64);
  assert(s.count(0) == 1);
  assert(s.count(63) == 1);
  assert(s.count(64) == 0);
  return 0;
}
