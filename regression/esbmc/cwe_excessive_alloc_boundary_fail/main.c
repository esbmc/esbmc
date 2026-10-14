#include <stdlib.h>

// A request of 100 bytes against a bound of 99 exceeds K by one byte and must
// be flagged. Together with cwe_excessive_alloc_boundary_ok this pins the
// exact `size <= K` boundary against a silent off-by-one regression.
int main(void)
{
  char *p = malloc(100);
  free(p);
  return 0;
}
