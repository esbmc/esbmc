// <charconv> is not covered by ESBMC's bundled OMs, but transitively pulls
// in the host's <utility>, whose std::pair collides with ESBMC's own
// std::pair (OMs live directly in namespace std, not an inline namespace,
// so this is a hard ambiguity, not an overload). This is the documented
// trade-off of --mix-cpp-host-headers, not a bug: this test pins that the
// ambiguity is still reported rather than silently misresolved.
#include <charconv>

int main()
{
  return 0;
}
