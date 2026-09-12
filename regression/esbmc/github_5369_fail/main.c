/* The counterpart to github_5369: the write lands *inside* the pointer's bytes,
 * so the two reads genuinely differ and the equality must still be refuted. */
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

  unsigned long j = nondet_ulong();
  if (j >= 8)
    return 0;
  buf[j] = 0;

  void *r2 = *(void **)buf;
  if (hit)
    assert(r1 == r2);
  return 0;
}
