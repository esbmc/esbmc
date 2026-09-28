// <ext/hash_map> is a pre-standard extension that ESBMC's bundled OMs do not
// cover, so under --mix-cpp-host-headers it falls through to the host header
// (libstdc++ and libc++ both ship it). The host version is built on its own
// library's internals, but ESBMC's OMs shadow the headers it includes and
// provide only the public interfaces. Mixing the two therefore fails to
// compile. This is the documented trade-off of --mix-cpp-host-headers, not a
// bug: this test pins that the collision is reported as a hard error rather
// than silently misresolved. A standard header would stop colliding once it
// is modelled, as <regex> did.
#include <ext/hash_map>

int main()
{
  return 0;
}
