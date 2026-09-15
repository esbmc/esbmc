// [stream.types]: streamoff must represent the largest file offset, and
// streamsize is a signed type. The model made both 32-bit, and streamsize
// unsigned, so offsets past 2 GiB and negative sizes did not survive (#7540).
// <fstream> parses only while ostream's seekp/tellp agree on the wider type.
#include <cassert>
#include <fstream>
#include <ios>

int main()
{
  std::streamoff big = 3000000000LL;
  assert(big == 3000000000LL);
  assert(sizeof(std::streamoff) == 8);

  std::streampos pos = big;
  std::ios::off_type off = big;
  assert(pos == 3000000000LL && off == big);

  std::streamsize n = -1;
  assert(n < 0);
  return 0;
}
