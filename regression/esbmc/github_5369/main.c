/* Two reads of one location through a pointer rebuilt from bytes must agree.
 * convert_typecast_to_ptr() matches the integer only against objects already in
 * addr_space_data, so the read before `&later` was converted could not reach
 * `later` and fell back to the invalid object, while the read after it could --
 * equal addresses, unequal pointers (#5369). */
#include <assert.h>
#include <stdlib.h>

unsigned char nondet_uchar(void);
unsigned long nondet_ulong(void);

int main(void)
{
  char *buf = malloc(16);
  if (!buf)
    return 0;
  for (int i = 0; i < 16; i++)
    buf[i] = nondet_uchar();

  void *r1 = *(void **)buf;
  int later;
  _Bool hit = (unsigned long)r1 == (unsigned long)&later;

  /* Write above the pointer's bytes: the reconstructed address cannot change. */
  unsigned long j = nondet_ulong();
  if (j < 8 || j >= 16)
    return 0;
  buf[j] = 0;

  void *r2 = *(void **)buf;
  if (hit)
    assert(r1 == r2);
  return 0;
}
