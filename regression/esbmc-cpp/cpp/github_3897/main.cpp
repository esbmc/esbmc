// <cfenv> is not covered by ESBMC's bundled OMs, so it is only reachable
// when the host system headers are made visible alongside them. It is a thin
// wrapper over <fenv.h>, which ESBMC's own libc does model, so mixing the two
// trees is enough to compile it.
#include <cfenv>

int main()
{
  __ESBMC_assert(
    FE_TONEAREST == FE_TONEAREST, "host-only header compiled alongside ESBMC's OMs");
  return 0;
}
