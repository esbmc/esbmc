#include <cassert>
#include <variant>

int main()
{
  std::variant<int, double> v = 7;
  // index is 0 (int); the assertion below must fail.
  assert(v.index() == 1);
  return 0;
}
