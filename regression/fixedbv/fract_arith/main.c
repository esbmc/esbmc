/* All expected values pinned by native execution (clang -ffixed-point).
 * s.7 values are exact dyadics k/128. Division and multiplication round
 * down (floor), matching Clang / llvm.sdiv.fix semantics. */
#include <assert.h>

int main(void)
{
  short _Fract a = -0.671875hr; /* raw 0xaa = -86/128 */
  short _Fract b = 0.9921875hr; /* raw 0x7f = 127/128 */

  assert(a + b == 0.3203125hr);      /* 41/128 */
  assert(b - 0.25hr == 0.7421875hr); /* 95/128 */
  assert(a / b == -0.6796875hr);     /* floor(-86*128/127) = -87 */
  assert(a * b == -0.671875hr);      /* floor(-86*127/128) = -86 */

  short _Fract v1 = -0.765625hr;   /* -98/128 */
  assert(v1 / b == -0.7734375hr);  /* -99/128 */
  short _Fract v2 = -0.03125hr;    /* -4/128 */
  assert(v2 / b == -0.0390625hr);  /* -5/128 */
  short _Fract v3 = -0.8671875hr;  /* -111/128 */
  assert(v3 / b == -0.875hr);      /* -112/128 */
  short _Fract v4 = -0.8046875hr;  /* -103/128 */
  short _Fract b2 = 0.984375hr;    /* 126/128 */
  assert(v4 / b2 == -0.8203125hr); /* -105/128 */

  assert(a < b);
  assert(b2 >= 0.984375hr);
  return 0;
}
