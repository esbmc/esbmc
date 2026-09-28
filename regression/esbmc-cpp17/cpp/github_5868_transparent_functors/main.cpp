#include <functional>
#include <cassert>

int main()
{
  // Transparent comparators compare at the operands' own types: binding
  // std::less<> to std::less<int> truncates both sides and answers wrongly.
  assert(std::less<>()(0.5, 0.7) == true);
  assert(std::greater<>()(0.7, 0.5) == true);
  assert(std::less_equal<>()(0.5, 0.5) == true);
  assert(std::greater_equal<>()(0.7, 0.5) == true);
  assert(std::equal_to<>()(0.5, 0.5) == true);
  assert(std::not_equal_to<>()(0.5, 0.7) == true);

  // ... and combine at their own types, rather than in int.
  assert(std::plus<>()(1.5, 2.25) == 3.75);
  assert(std::minus<>()(2.5, 0.25) == 2.25);
  assert(std::multiplies<>()(1.5, 2.0) == 3.0);
  assert(std::divides<>()(5.0, 2.0) == 2.5);
  assert(std::modulus<>()(7, 4) == 3);
  assert(std::negate<>()(0.5) == -0.5);

  assert(std::logical_and<>()(true, false) == false);
  assert(std::logical_or<>()(true, false) == true);
  assert(std::logical_not<>()(false) == true);

  assert(std::bit_and<>()(6, 3) == 2);
  assert(std::bit_or<>()(6, 3) == 7);
  assert(std::bit_xor<>()(6, 3) == 5);
  assert(std::bit_not<>()(0) == -1);

  // Pointer comparison is what motivated is_transparent: less<int> cannot do
  // it at all. Same array, so the relational comparison is well defined.
  int a[2] = {1, 2};
  assert(std::less<>()(&a[0], &a[1]) == true);

  // Heterogeneous operands.
  assert(std::less<>()(1, 2.5) == true);
  return 0;
}
