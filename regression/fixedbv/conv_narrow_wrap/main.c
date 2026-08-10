/* Out-of-range narrowing into a NON-_Sat fixed type keeps the destination's
 * low bits (it does not clamp -- that is what _Sat is for). Values pinned by
 * native execution: -3.5 * 128 = -448, low 8 bits = 0x40 = +0.5.
 *
 * The constant folder must agree with the encoder here: BigInt's % truncates
 * toward zero rather than wrapping, which used to fold this to a value
 * outside its own format and flip the verdict against --no-simplify. */
#include <assert.h>

int main(void)
{
  short _Fract n = (short _Fract)(-3.5k);
  assert(n == 0.5hr);

  short _Fract m = (short _Fract)(2.5k); /* 320 -> 320 - 256 = 64 */
  assert(m == 0.5hr);

  union
  {
    short _Fract f;
    signed char r;
  } p;
  p.f = n;
  assert(p.r == 0x40);
  return 0;
}
