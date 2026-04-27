#include <cassert>

namespace ns {
enum class E : int
{
  A = 1
};
}  // namespace ns

using ns::E;  // using-declaration of an enum-class type

int main()
{
  assert(static_cast<int>(E::A) == 1);
  return static_cast<int>(E::A);
}
