// setprecision(-1) leaves precision() negative, and a negative precision
// formats as the default six digits. The model clamped it to 0, the reason
// being an unsigned streamsize, and printed only the integer part (#7540).
#include <cassert>
#include <cstring>
#include <iomanip>
#include <sstream>

int main()
{
  std::ostringstream s;
  s << std::setprecision(-1);
  assert(s.precision() < 0);
  s << 1.5;
  assert(strcmp(s.str().c_str(), "1.5") == 0);
  return 0;
}
