// Storing a non-trivial element must construct into the raw slot, not assign
// into it: assignment runs the element's operator=, which frees the garbage
// pointer an uninitialised slot holds.
#include <vector>
#include <cassert>

int main()
{
  std::vector<std::vector<int>> v;
  std::vector<int> row;
  row.push_back(5);

  v.push_back(row);

  assert(v.size() == 1);
  assert(v[0].size() == 1);
  assert(v[0][0] == 5);
  return 0;
}
