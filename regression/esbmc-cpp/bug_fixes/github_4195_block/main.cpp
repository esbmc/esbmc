#include <cassert>

namespace ns {
enum class Color
{
  Red,
  Green,
  Blue
};
}  // namespace ns

int main()
{
  using enum ns::Color;  // block-scope C++20 'using enum'
  assert(static_cast<int>(Green) == 1);
  return static_cast<int>(Red);
}
