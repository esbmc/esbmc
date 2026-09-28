#include <cassert>
#include <set>

int main()
{
  // set is ordered and unique: {3,1,2,1} holds 1,2,3 and begins at 1.
  std::set<int> s{3, 1, 2, 1};
  assert(s.size() == 4);

  return 0;
}
