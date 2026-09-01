// Once reserve() has run, push_back must not re-enter its loop: the cost of
// this program is independent of --unwind: 831 VCCs at 40, 80 and 120.
// Before reserve() reallocated in place it was 9076, 17236 and 25396.
#include <vector>
#include <cassert>

int main()
{
  std::vector<int> v;
  v.reserve(20);
  for (int i = 0; i < 8; i++)
    v.push_back(i);
  assert(v[0] == 0);
  assert(v[7] == 7);
  assert(v.size() == 8);
  return 0;
}
