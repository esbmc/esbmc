// #7797: <execution> must be includable and carry the policy tags.
// Validated against libc++ with -D_LIBCPP_ENABLE_EXPERIMENTAL, which is how it
// exposes the policies; libstdc++ needs no macro.
#include <execution>
#include <cassert>
#include <type_traits>

int main()
{
  assert(std::is_execution_policy<std::execution::sequenced_policy>::value);
  assert(std::is_execution_policy<std::execution::parallel_policy>::value);
  assert(std::is_execution_policy_v<std::execution::parallel_unsequenced_policy>);
  assert(!std::is_execution_policy<int>::value);
  assert(!std::is_execution_policy_v<double>);

  assert((std::is_same<
          decltype(std::execution::seq),
          const std::execution::sequenced_policy>::value));
  assert((std::is_same<
          decltype(std::execution::par_unseq),
          const std::execution::parallel_unsequenced_policy>::value));
  return 0;
}
