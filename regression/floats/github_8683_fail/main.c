/* github #8683: each assertion contradicts C11 7.12.7.3 or F.10.4.3 */
#include <assert.h>
#include <math.h>
int main(void)
{
  assert(isinf(hypot(1e200, 1e200)));
  assert(hypot(1e-200, 1e-200) == 0.0);
  assert(isnan(hypot(INFINITY, NAN)));
  return 0;
}
