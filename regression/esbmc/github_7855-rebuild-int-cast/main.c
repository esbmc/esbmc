// esbmc/esbmc#7855: an address rebuilt as a pointer may name an object
// allocated later; read back as an integer before that allocation, it must
// still be the address.
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

  __ESBMC_assert(back == v, "an address read back as a pointer keeps its value");
  return 0;
}
