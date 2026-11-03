#include <assert.h>
#include <string.h>

/* On a big-endian target the byte at the lowest address is the most
 * significant one, so a partial memcpy between scalars moves their top
 * bytes. */
int main()
{
  unsigned y = 0x01020304u;

  unsigned x = 0;
  memcpy(&x, &y, 1);
  assert(x == 0x01000000u);

  unsigned z = 0;
  memcpy((char *)&z + 2, (char *)&y, 1);
  assert(z == 0x00000100u);

  unsigned w = 0;
  memcpy((char *)&w, (char *)&y + 1, 2);
  assert(w == 0x02030000u);
  return 0;
}
