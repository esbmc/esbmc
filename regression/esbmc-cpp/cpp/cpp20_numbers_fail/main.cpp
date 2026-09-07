// [numbers.syn]: the C++20 mathematical constants. Values checked against
// g++ -std=c++20 at runtime, not just for parseability.
#include <numbers>
#include <cassert>

int main()
{
  assert(std::numbers::pi > 3.14159 && std::numbers::pi < 3.14000);
  assert(std::numbers::e > 2.71828 && std::numbers::e < 2.71829);
  assert(std::numbers::sqrt2 > 1.41421 && std::numbers::sqrt2 < 1.41422);
  assert(std::numbers::ln2 < std::numbers::ln10);
  assert(std::numbers::pi_v<float> > 3.14f);
  return 0;
}
