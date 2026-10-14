// An offset of 3000000000 is positive. A 32-bit streamoff wrapped it negative
// and verified this assertion (#7540).
#include <cassert>
#include <ios>

int main()
{
  std::streamoff big = 3000000000LL;
  assert(big < 0);
  return 0;
}
