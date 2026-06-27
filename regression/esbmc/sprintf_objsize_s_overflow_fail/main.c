#include <limits.h>
extern int sprintf(char *restrict s, const char *restrict fmt, ...);

// Companion to sprintf_objsize_s_no_overflow: the object-size bound must be
// SOUND, never an under-approximation.  With a 64-byte source the return can
// be as large as 63, so `base + n` with base = INT_MAX - 10 has a reachable
// overflow.  Guards against re-introducing the "model an unknown %s as 0"
// behaviour, which would silently mask this overflow.
int main(void)
{
  char src[64]; // nondet content: strlen can reach 63
  char dst[128];
  int n = sprintf(dst, "%s", src);
  int base = INT_MAX - 10;
  int sum = base + n; // n up to 63 > 10: overflow reachable
  return sum;
}
