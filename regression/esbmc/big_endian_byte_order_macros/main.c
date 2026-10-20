#include <assert.h>

/* --big-endian on a little-endian target must change the byte-order macros
   too, or this picks the little-endian expectation. */
int main()
{
  unsigned x = 0x01020304u;
  unsigned char *p = (unsigned char *)&x;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__ && defined(__BIG_ENDIAN__) &&      \
  !defined(__LITTLE_ENDIAN__)
  assert(p[0] == 1);
#else
  assert(p[0] == 4);
#endif
  return 0;
}
