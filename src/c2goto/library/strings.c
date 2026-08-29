#if defined(__unix__) || defined(__APPLE__) || defined(__MACH__)
#  include <strings.h>
#endif

/* Each of these scans on the unsigned counterpart: right-shifting a negative
 * signed value is implementation-defined (C11 6.5.7p5), and the conversion is
 * value-preserving on the bit pattern the scan is looking at. */

int ffs(int x)
{
__ESBMC_HIDE:;
  if (x == 0)
    return 0;

  unsigned v = (unsigned)x;
  int pos = 1;
  while ((v & 1) == 0)
  {
    v >>= 1;
    pos++;
  }

  return pos;
}

int ffsl(long x)
{
__ESBMC_HIDE:;
  if (x == 0)
    return 0;

  unsigned long v = (unsigned long)x;
  int pos = 1;
  while ((v & 1) == 0)
  {
    v >>= 1;
    pos++;
  }

  return pos;
}

int ffsll(long long x)
{
__ESBMC_HIDE:;
  if (x == 0)
    return 0;

  unsigned long long v = (unsigned long long)x;
  int pos = 1;
  while ((v & 1) == 0)
  {
    v >>= 1;
    pos++;
  }

  return pos;
}
