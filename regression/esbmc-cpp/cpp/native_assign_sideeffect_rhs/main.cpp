#include <cassert>
#include <vector>

// Exercises the assignment shape the census observed at this handler: an rhs
// that is a side-effecting expression, reached through the container
// operational models. Delegating the statement to convert_assign must preserve
// the stored values and the size.
int main()
{
  std::vector<int> v;
  v.push_back(3);
  v.push_back(4);

  int first = v[0];
  int second = v[1];

  assert(v.size() == 2);
  assert(first == 3);
  assert(second == 4);
  return 0;
}
