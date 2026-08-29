// Anti-vacuity twin of placeholders_distinct_types: is_placeholder reports the
// placeholder's own index, so it cannot also report a different one.
#include <functional>
#include <type_traits>
#include <cassert>

int main()
{
  assert(std::is_placeholder<std::decay<decltype(std::placeholders::_3)>::type>::value == 4);
  return 0;
}
