// reserve() reallocates in place, so a non-trivial element's own heap buffer
// must survive the growth. The elements here cross the default capacity of 10,
// which the github_6368 tests do not: they store a single element, so
// reserve()'s body never runs.
#include <vector>
#include <cassert>

int main()
{
  std::vector<std::vector<int> > v;
  for (int i = 0; i < 12; i++)
  {
    std::vector<int> row;
    row.push_back(i);
    v.push_back(row);
  }
  assert(v.size() == 12);
  assert(v[0].size() == 1);
  assert(v[0][0] == 0);
  assert(v[11][0] == 11);
  return 0;
}
