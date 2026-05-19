#include <assert.h>

int main(void)
{
  int x;
  if (x > 0)
    x--;
  else
    x++;
  // Trivially true: no enumeration should occur on the UNSAT path.
  assert(x == x);
  return 0;
}
