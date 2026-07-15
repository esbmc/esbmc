#include <stdlib.h>
#include <assert.h>

// strtol's hex-digit conversion applied tolower() and then an offset (-7)
// that is only correct for uppercase input, so 'a'..'f' decoded to garbage
// (e.g. 'a' -> 42 instead of 10). Caught while adding the sibling strtoll
// operational model in the same code (github issue #1016).
int main()
{
  long v1 = strtol("2A", NULL, 16);
  assert(v1 == 42);

  long v2 = strtol("2a", NULL, 16);
  assert(v2 == 42);

  long v3 = strtol("ff", NULL, 16);
  assert(v3 == 255);

  return 0;
}
