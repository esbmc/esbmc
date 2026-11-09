#include <string.h>

/* The elements memset zeroed no longer point at x, so the read is through
 * NULL. */
int x;

int main()
{
  int *a[4] = {&x, &x, &x, &x};
  memset(a, 0, sizeof a);
  return *a[2];
}
