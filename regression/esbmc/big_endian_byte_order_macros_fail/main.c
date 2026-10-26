#include <assert.h>

/* As big_endian_byte_order_macros, with the expectations swapped. */
int main()
{
  unsigned x = 0x01020304u;
  unsigned char *p = (unsigned char *)&x;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__ && defined(__BIG_ENDIAN__) &&      \
  !defined(__LITTLE_ENDIAN__)
  assert(p[0] == 4);
#else
  assert(p[0] == 1);
#endif
  return 0;
}
