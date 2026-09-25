// ios::copyfmt was declared and never defined, so it returned a reference to
// nothing and copied no state. [basic.ios.members]: it copies the formatting
// state (flags, precision, width, fill, exception mask), not the buffer.
#include <sstream>
#include <cassert>

int main()
{
  std::ostringstream a, b;
  a.precision(7);
  a.fill('*');
  b.copyfmt(a);
  assert(b.precision() == 7);
  assert(b.fill() == '*');
  return 0;
}
