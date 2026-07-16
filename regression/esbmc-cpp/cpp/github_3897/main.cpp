// <version> is a macro-only libc++/libstdc++ header with no operational
// model in ESBMC's bundled C++ library, so it is only reachable when the
// host system headers are made visible alongside the OMs.
#include <version>

int main()
{
  __ESBMC_assert(1, "host-only header compiled alongside ESBMC's OMs");
  return 0;
}
