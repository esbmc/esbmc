#include <cmath>
#include <cassert>

int main()
{
  int quo = 0;
  // remquo yields the remainder, not the quotient.
  assert(std::remquo(10.0, 3.0, &quo) == 3.0);
  return 0;
}
