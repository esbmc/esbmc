#include <sstream>
#include <iomanip>
#include <cstring>
#include <cassert>
// [ios.base]: a negative width is well-formed and means "no padding". The
// model's streamsize is unsigned, so it has to clamp rather than wrap.
int main()
{
  std::ostringstream s;
  s << std::setw(-1) << 5;
  assert(strcmp(s.str().c_str(), "5") == 0);
  return 0;
}
