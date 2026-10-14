// Negative variant of OM_sanity_checks/iosfwd (esbmc/esbmc#5868, Gap 6a):
// proves the <iosfwd> forward declarations coexist with the concrete stream
// definitions -- pulling in <iostream>/<fstream>/<sstream> brings the
// definitions of all 14 forward-declared classes into the TU, so a redefinition
// mismatch would surface as a PARSE ERROR here rather than the expected
// VERIFICATION FAILED. The false assertion proves ESBMC verifies the TU.
#include <iosfwd>
#include <iostream>
#include <fstream>
#include <sstream>

int main()
{
  int x = 5;
  __ESBMC_assert(x == 6, "intentionally false");
  return 0;
}
