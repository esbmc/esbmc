#include <stdlib.h>

// A bare malloc(sizeof(T)) is lowered to element-count 1 with element type T
// (never a raw byte count), so the excessive-size check must scale by
// sizeof(T). Here sizeof(struct Huge) is 2 MiB > the default 1 MiB bound, so
// the allocation must be flagged CWE-789. Pins the typed-size scaling fix:
// without it the check would compare 1 <= K and pass silently.
struct Huge
{
  char b[2 * 1024 * 1024];
};

int main(void)
{
  struct Huge *p = malloc(sizeof(struct Huge));
  free(p);
  return 0;
}
