// [iosfwd.syn] declares streamoff and streampos at namespace scope; the model
// had them only inside class ios, so the standard spelling did not parse
// (#7540). Their width is unchanged and still not standard-conforming.
#include <cassert>
#include <ios>

std::streamoff offset = 0;
std::streampos position = 0;

int main()
{
  assert(offset == 0);
  assert(position == 0);

  std::ios::pos_type p = 0;
  std::ios::off_type o = 0;
  assert(p == 0 && o == 0);

  std::streamoff moved = offset + 7;
  assert(moved == 7);
  return 0;
}
