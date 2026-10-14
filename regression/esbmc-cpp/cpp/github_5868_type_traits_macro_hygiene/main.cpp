// A user is free to define any non-reserved identifier as a macro, so the OM
// headers must name template parameters in the implementation's reserved space.
// remove_all_extents spelled its extent parameter `N`, which this #define
// rewrote to a literal, making the partial specialization unusable (#7337).
#define N 2

#include <type_traits>
#include <cassert>

static_assert(std::is_same<std::remove_all_extents_t<int[N]>, int>::value, "");
static_assert(std::is_same<std::remove_all_extents_t<int[N][3]>, int>::value, "");
static_assert(std::is_same<std::remove_all_extents_t<char[4][5]>, char>::value, "");

int main()
{
  assert(sizeof(std::remove_all_extents_t<int[N][3]>) == sizeof(int));
  return 0;
}
