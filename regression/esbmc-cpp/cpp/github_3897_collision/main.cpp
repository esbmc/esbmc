// <regex> is not covered by ESBMC's bundled OMs, so under
// --mix-cpp-host-headers it falls through to the host header. The host
// <regex> is built on its own library's <string>, <vector> and <locale>
// internals, but ESBMC's OMs shadow those headers and provide only the
// public interfaces. Mixing the two therefore fails to compile. This is the
// documented trade-off of --mix-cpp-host-headers, not a bug: this test pins
// that the collision is reported as a hard error rather than silently
// misresolved.
#include <regex>

int main()
{
  return 0;
}
