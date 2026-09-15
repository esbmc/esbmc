// <ratio> was not found although <chrono> modelled std::ratio (github #7797).
#include <ratio>
#include <cassert>

using Sum = std::ratio_add<std::atto, std::atto>;
using Diff = std::ratio_subtract<std::milli, std::micro>;
using Half = std::ratio<9223372036854775807, 2>;
using Big = std::ratio_add<Half, Half>;
using Product = std::ratio_multiply<std::exa, std::atto>;
using Quotient = std::ratio_divide<std::kilo, std::mega>;
using AttoLessExa = std::ratio_less<std::atto, std::exa>;
using MilliGeMilli = std::ratio_greater_equal<std::milli, std::milli>;
using MinusThirdEqHalf = std::ratio_equal<std::ratio<-1, 3>, std::ratio<1, -2>>;

static_assert(Sum::num == 1 && Sum::den == 500000000000000000, "reduced");
static_assert(Big::num == 9223372036854775807 && Big::den == 1, "no overflow");

using Zero = std::ratio_add<std::ratio<0>, std::ratio<0>>;
static_assert(Zero::num == 0 && Zero::den == 1, "sum of zero ratios");

// One case per branch of the continued-fraction comparison, and both sign
// mixes.
static_assert(std::ratio_less<std::ratio<1>, std::ratio<3, 2>>::value, "");
static_assert(!std::ratio_less<std::ratio<3, 2>, std::ratio<1>>::value, "");
static_assert(!std::ratio_less<std::ratio<1, 2>, std::ratio<1, 3>>::value, "");
static_assert(
  std::ratio_less<std::ratio<21, 13>, std::ratio<13, 8>>::value,
  "");
static_assert(
  !std::ratio_less<std::ratio<13, 8>, std::ratio<21, 13>>::value,
  "");
static_assert(!std::ratio_less<std::ratio<2>, std::ratio<2>>::value, "");
static_assert(std::ratio_less<std::ratio<-1, 2>, std::ratio<-1, 3>>::value, "");
static_assert(std::ratio_less<std::ratio<-1, 2>, std::ratio<1, 3>>::value, "");
static_assert(!std::ratio_less<std::ratio<1, 3>, std::ratio<-1, 2>>::value, "");

int main()
{
  assert(Diff::num == 999 && Diff::den == 1000000);
  assert(Product::num == 1 && Product::den == 1);
  assert(Quotient::num == 1 && Quotient::den == 1000);
  assert(AttoLessExa::value);
  assert(MilliGeMilli::value);
  assert(!MinusThirdEqHalf::value);
  return 0;
}
