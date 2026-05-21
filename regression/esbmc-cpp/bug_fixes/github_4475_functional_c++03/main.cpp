// Pin issue #4475: <functional> must parse and verify cleanly under --std c++03.
// Exercises the comparator/arithmetic/logical/bitwise functors that stay
// available in both modes via the OM_CONSTEXPR macro.
#include <cassert>
#include <functional>

int main()
{
  std::less<int> lt;
  std::greater<int> gt;
  std::equal_to<int> eq;
  std::not_equal_to<int> ne;
  std::less_equal<int> le;
  std::greater_equal<int> ge;
  assert(lt(1, 2));
  assert(gt(2, 1));
  assert(eq(3, 3));
  assert(ne(3, 4));
  assert(le(3, 3));
  assert(ge(3, 3));

  std::plus<int> add;
  std::minus<int> sub;
  std::multiplies<int> mul;
  std::divides<int> div;
  std::modulus<int> mod;
  std::negate<int> neg;
  assert(add(2, 3) == 5);
  assert(sub(5, 2) == 3);
  assert(mul(2, 3) == 6);
  assert(div(6, 2) == 3);
  assert(mod(7, 3) == 1);
  assert(neg(4) == -4);

  std::logical_and<bool> land;
  std::logical_or<bool> lor;
  std::logical_not<bool> lnot;
  assert(land(true, true));
  assert(lor(false, true));
  assert(lnot(false));

  std::bit_and<int> band;
  std::bit_or<int> bor;
  std::bit_xor<int> bxor;
  std::bit_not<int> bnot;
  assert(band(0xF0, 0x0F) == 0x00);
  assert(bor(0xF0, 0x0F) == 0xFF);
  assert(bxor(0xFF, 0x0F) == 0xF0);
  assert(bnot(0) == ~0);

  return 0;
}
