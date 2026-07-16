// Same as regression/esbmc-cpp/cpp/github_3897, but run WITHOUT
// --mix-cpp-host-headers: <version> has no operational model, so the
// default (opt-out) behaviour must still reject it with a parse error,
// proving --mix-cpp-host-headers is genuinely opt-in.
#include <version>

int main()
{
  __ESBMC_assert(1, "host-only header should not be reachable by default");
  return 0;
}
