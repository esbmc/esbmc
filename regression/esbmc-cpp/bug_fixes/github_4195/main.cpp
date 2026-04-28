#include <cassert>

namespace ns {
enum class E
{
  A,
  B
};
}  // namespace ns

using enum ns::E;  // C++20: brings A and B into the enclosing scope

int main()
{
  assert(static_cast<int>(A) == 0);
  assert(static_cast<int>(B) == 1);
  return static_cast<int>(A);
}
