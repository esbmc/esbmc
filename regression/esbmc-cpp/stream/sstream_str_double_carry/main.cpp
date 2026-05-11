// Half-up rounding in operator<<(double) can produce a fractional integer
// equal to the precision multiplier (e.g. 0.999999 * 1e6 + 0.5 -> 1e6),
// which would silently drop the leading carry digit if the model only
// reserved `prec` characters for the fractional buffer. The carry must be
// promoted into the integer part instead.
#include <sstream>
#include <cassert>
using namespace std;

int main()
{
  stringstream oss;
  double v = 1.999999;
  oss << v;
  assert(oss.str() == "2");
  return 0;
}
