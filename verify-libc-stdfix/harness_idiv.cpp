// Verify LLVM libc's fixed_point::idiv against TR 18037 7.18a.6.7: idivfx
// computes x/y as an integer, rounded toward zero.
//
// Both operands share a format, so the quotient of the VALUES equals the
// quotient of their scaled raw representations -- the property is just that
// idiv agrees with C's truncating integer division on the raws.
//
// One input is excluded: x = format minimum, y = -1 ulp. There the exact
// quotient is 2^(N-1), which does not fit the signed N-bit CompType the
// implementation divides in, so the result wraps. TR 18037 says "if an
// integer result of one of these functions overflows, the behavior is
// undefined", and LLVM libc cites exactly that -- but the overflow is in the
// intermediate, not in the `int` return type, which holds the value fine.
// Whether that is UB-excused is a question for the library, not something
// this harness should assert either way. See RESULTS.md.
#include "src/__support/fixed_point/fx_bits.h"

extern "C" short _Fract nondet_sfract8();
extern "C" void __ESBMC_assert(bool, const char *);
extern "C" void __ESBMC_assume(bool);

using LIBC_NAMESPACE::fixed_point::idiv;

int main()
{
  short _Fract x = nondet_sfract8();
  short _Fract y = nondet_sfract8();

  union Pun
  {
    short _Fract f;
    signed char raw;
  };
  Pun px, py;
  px.f = x;
  py.f = y;

  __ESBMC_assume(py.raw != 0);                    // divide by zero is UB
  __ESBMC_assume(!(px.raw == -128 && py.raw == -1)); // intermediate overflow

  int got = idiv<short _Fract, int>(x, y);
  int want = px.raw / py.raw; // C truncates toward zero

  __ESBMC_assert(got == want, "idiv(x,y) == raw_x / raw_y (toward zero)");
  return 0;
}
