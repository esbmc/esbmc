/* Float -> fixed rounds TOWARD ZERO, which differs from fixed -> fixed
 * narrowing (floor). Both directions pinned by native execution. */
#include <assert.h>

int main(void)
{
  assert((_Accum)1.5f == 1.5k);
  assert((_Fract)0.5f == 0.5r);
  assert((_Fract)-0.5f == -0.5r);

  /* 3.7 is not representable: toward zero keeps the magnitude below it on
   * both signs, so the negative case is NOT the floor. */
  assert((_Accum)3.7f == 3.69998168945312500k);
  assert((_Accum)-3.7f == -3.69998168945312500k);

  /* half an ulp below zero truncates to 0; floor would give -1 ulp */
  assert((_Fract)(-1.0f / 65536.0f) == 0.0r);
  return 0;
}
