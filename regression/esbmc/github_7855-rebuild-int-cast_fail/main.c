// esbmc/esbmc#7855: the address read back is that one, not another.
#include <stdlib.h>

int main(void)
{
  unsigned char *buf = malloc(sizeof(void *));
  if (!buf)
    return 0;

  unsigned long v = nondet_ulong();
  for (unsigned i = 0; i < sizeof v; ++i)
    buf[i] = ((unsigned char *)&v)[i];
  void *p = *(void **)buf;
  unsigned long back = (unsigned long)p;
  char *later = malloc(16);
  if (!later)
    return 0;

  __ESBMC_assert(back == v + 1, "the address read back is another");
  return 0;
}
