#include <assert.h>

int main()
{
  int r = 0;
  for (int pass = 0; pass < 2; ++pass)
  {
    if (pass == 1)
      goto WAIT;
    {
      int i = 5;
      continue;
    WAIT:
      r = i;
    }
  }
  // The jump skips i's declaration, so i is indeterminate (C11 6.2.4p6)
  assert(r == 5);
}
