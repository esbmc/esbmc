// Issue #4240: Under C++20 [expr.shift]/2, signed left-shift result overflow
// is defined as wrapping (E1 * 2^E2 mod 2^N). --overflow-check must NOT flag
// this as a bug even when the result overflows int.
#include <cstdint>

extern "C" uint8_t nondet_u8();

int main()
{
  uint8_t b = nondet_u8();
  __ESBMC_assume(b >= 128); // forces the wrapping path
  volatile int r = (int)b << 24;
  (void)r;
}
