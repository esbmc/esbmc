// <cfenv>, <cinttypes>, <cwctype>, <cuchar>, <uchar.h>, <cstdbool>,
// <cstdalign>, <ctgmath> and <ccomplex> were not found: -nostdinc++ removes
// the host library's wrappers and the models had none (github #7795).
#include <cfenv>
#include <cinttypes>
#include <cwctype>
#include <cuchar>
#include <uchar.h>
#include <cstdbool>
#include <cstdalign>
#include <ctgmath>
#include <ccomplex>
#include <cassert>

int main()
{
  assert(std::fesetround(FE_UPWARD) == 0);
  assert(std::fegetround() == FE_UPWARD);

  std::intmax_t m = std::imaxabs(-3);
  assert(m == 3);

  std::wint_t w = L'a';
  (void)w;
  (void)std::iswalpha;

  std::mbstate_t state = std::mbstate_t();
  (void)state;

  assert(__bool_true_false_are_defined == 1);
  assert(__alignas_is_defined == 1);

  std::complex<double> z(1.0, 2.0);
  assert(z.real() == 1.0 && z.imag() == 2.0);
  return 0;
}
