/* Built with goto-cc --big-endian: the union initialiser is a
 * byte_update_big_endian and the bit cast a byte_extract_big_endian (#8238). */
#include <assert.h>

struct T
{
  int a;
  int b;
};

union X
{
  struct T s;
  unsigned int f;
};

union X gx = {.f = 0x01020304u};

int main(void)
{
  float f = 2.0f;
  int k = __builtin_bit_cast(int, f);
  assert(k == 0x40000000);
  assert(gx.s.a == 0x01020304);
  assert(((unsigned char *)&gx)[0] == 1);
  return 0;
}
