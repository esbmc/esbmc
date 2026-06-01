#include <cassert>

namespace ns {
enum class E
{
  A,
  B
};
}  // namespace ns

using enum ns::E;

int main()
{
  // A is the 0th enumerator, so this assertion must fail.
  assert(static_cast<int>(A) == 1);
  return 0;
}
