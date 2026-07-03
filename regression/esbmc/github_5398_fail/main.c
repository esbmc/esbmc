#include <stdlib.h>

// Soundness guard for the issue #5398 fix: the malloc-zero-is-null change must
// not blunt genuine invalid-free detection. A double free of a valid dynamic
// object must still be reported, under the same flags as github_5398.
int main(void)
{
  void *p = malloc(4);
  free(p);
  free(p);
  return 0;
}
