#if defined(__unix__) || defined(__APPLE__) || defined(__MACH__)
#  include <strings.h>
#endif

int ffs(int x)
{
// cppcheck-suppress unusedLabel
__ESBMC_HIDE:;
  if (x == 0)
    return 0;

  int pos = 1;
  while ((x & 1) == 0)
  {
    x >>= 1;
    pos++;
  }

  return pos;
}
