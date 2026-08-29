// Negative twin of github_6368: the element must actually be stored, so a
// wrong-value assertion has to be reported rather than passing vacuously.
#include <vector>
#include <cassert>

int main()
{
  std::vector<std::vector<int>> v;
  std::vector<int> row;
  row.push_back(5);

  v.push_back(row);

  assert(v[0][0] == 6);
  return 0;
}
