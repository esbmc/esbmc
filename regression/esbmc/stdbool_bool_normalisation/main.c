#include <assert.h>
#include <stdbool.h>

int main(void)
{
  unsigned flags = 0x4u;

  /* C11 6.3.1.2: converting to _Bool yields 0 or 1, not the source value. */
  bool found = flags & 0x4u;
  assert(found == true);
  assert(found == 1);

  bool a = (bool)2, b = (bool)3;
  assert(a == b);

  /* C11 7.18: bool expands to _Bool, so it is the one-byte boolean type. */
  assert(sizeof(bool) == sizeof(_Bool));

  unsigned char raw;
  __ESBMC_assume(raw != 0);
  bool nz = raw;
  assert(nz == true);

  return 0;
}
