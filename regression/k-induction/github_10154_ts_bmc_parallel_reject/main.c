// #10154: the inductive step assumed that the first loop is entered, and with
// n == 0 it never is, so every path ended in abort() and the bug in the third
// loop was reported as proved.
#include <assert.h>
#include <stdlib.h>

int main(void)
{
  int n = 0;
  int v, x;
  for (x = 0; x < n; x++)
    abort();
  for (v = 0; v < 4; v++)
    n = n + 1;
  for (x = 0; x < n; x++)
    assert(x < 2);
  return 0;
}
