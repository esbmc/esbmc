#include <stdlib.h>

// A request of exactly K bytes must pass: the bound is `size <= K`, so K is
// allowed and only K+1 is flagged. Pins the off-by-one direction of the check.
int main(void)
{
  char *p = malloc(100);
  free(p);
  return 0;
}
