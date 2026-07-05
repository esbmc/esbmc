#include <limits.h>
extern int sprintf(char *restrict s, const char *restrict fmt, ...);

// A non-literal %s whose argument points into a known-size array bounds the
// formatted length: strlen(src) <= sizeof(src)-1 = 15, so n <= 6 + 15 = 21.
// Before the object-size bound the return of a runtime-filled array was
// under-approximated to 0 (and an unknown array to unbounded); now it is a
// sound, finite over-approximation, so `base + n` is provably safe.
int main(void)
{
  char src[16]; // nondet content: strlen <= 15
  char dst[128];
  int n = sprintf(dst, "value=%s", src);
  int base = INT_MAX - 100;
  int sum = base + n; // n <= 21, so base + n < INT_MAX: no overflow
  return sum;
}
