#include <stdlib.h>

// Soundness guard for #5398: a genuine double free must still be reported
// under the same flags as github_5398.
int main(void)
{
  void *p = malloc(4);
  free(p);
  free(p);
  return 0;
}
