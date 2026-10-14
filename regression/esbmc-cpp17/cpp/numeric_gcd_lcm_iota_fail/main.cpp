// Non-vacuity guard for numeric_gcd_lcm_iota: gcd really computes, so a wrong
// expectation must FAIL.
#include <numeric>
#include <cassert>

int main()
{
  assert(std::gcd(12, 18) == 7);
  return 0;
}
