#include <type_traits>
#include <cassert>

// [meta.trans.other] type_identity is C++20 (P0887R1); libc++ gates it the same
// way, so the c++17 sibling of this test pins the rejection.
static_assert(std::is_same<std::type_identity_t<int>, int>::value, "");
static_assert(std::is_same<std::type_identity_t<const int &>, const int &>::value, "");

int main()
{
  std::type_identity_t<int> x = 7;
  assert(x == 7);
  return 0;
}
