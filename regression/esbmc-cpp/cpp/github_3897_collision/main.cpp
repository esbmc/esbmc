// <charconv> is not covered by ESBMC's bundled OMs, so under
// --mix-cpp-host-headers it falls through to the host header. The host
// <charconv> depends on libstdc++ internals such as std::__bit_width, but
// ESBMC's OM <bit> shadows the host <bit> and provides only the public
// bit_width, not the __-prefixed internals. Mixing the two therefore fails
// to compile. This is the documented trade-off of --mix-cpp-host-headers,
// not a bug: this test pins that the collision is reported as a hard error
// rather than silently misresolved.
#include <charconv>

int main()
{
  return 0;
}
