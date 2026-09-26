/* Wide integer <-> fixed conversions; pinned by native execution. */
#include <assert.h>

int main(void)
{
  long long big = 70000;
  assert((long _Accum)big == 70000.0lk);
  assert((_Sat short _Accum)big == 255.9921875hk); /* clamps */

  long _Accum back = -70000.25lk;
  assert((long long)back == -70000); /* toward zero */
  assert((int)back == -70000);
  return 0;
}
