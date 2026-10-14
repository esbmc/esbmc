// <numeric> shipped accumulate/inner_product/partial_sum/adjacent_difference
// but not iota (C++11) or gcd/lcm/reduce (C++17), so naming them failed with
// "no member named 'gcd' in namespace 'std'".
//
// The sign and zero cases are the load-bearing part: [numeric.ops.gcd] and
// [numeric.ops.lcm] are defined on the absolute values, gcd(0,0) is 0, and lcm
// is 0 if either operand is 0.
#include <numeric>
#include <vector>
#include <cassert>

int main()
{
  assert(std::gcd(12, 18) == 6);
  assert(std::gcd(17, 5) == 1);
  assert(std::gcd(0, 0) == 0);
  assert(std::gcd(0, 5) == 5);
  assert(std::gcd(5, 0) == 5);
  assert(std::gcd(-12, 18) == 6);
  assert(std::gcd(12, -18) == 6);
  assert(std::gcd(-12, -18) == 6);

  assert(std::lcm(4, 6) == 12);
  assert(std::lcm(21, 6) == 42);
  assert(std::lcm(0, 5) == 0);
  assert(std::lcm(5, 0) == 0);
  assert(std::lcm(-4, 6) == 12);
  assert(std::lcm(4, -6) == 12);

  std::vector<int> v(3);
  std::iota(v.begin(), v.end(), 1);
  assert(v[0] == 1 && v[1] == 2 && v[2] == 3);

  assert(std::reduce(v.begin(), v.end(), 0) == 6);
  assert(std::reduce(v.begin(), v.end(), 10) == 16);

  // still present, unchanged
  assert(std::accumulate(v.begin(), v.end(), 0) == 6);

  return 0;
}
