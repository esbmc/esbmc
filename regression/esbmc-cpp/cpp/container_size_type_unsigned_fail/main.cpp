#include <vector>
#include <cassert>

int main()
{
  std::vector<int> v;
  // Wrong: the wrapped value is huge, not negative-ish small.
  assert(v.size() - 1 < 1000);
  return 0;
}
