// Negative counterpart of github_3387_array_cxx11: the arrays differ in the
// last element, so the equality must not hold.
#include <array>
#include <cassert>

int main()
{
  std::array<int, 3> a = {{1, 2, 3}};
  std::array<int, 3> c = {{1, 2, 4}};

  assert(a == c);

  return 0;
}
