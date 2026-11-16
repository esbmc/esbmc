#include <assert.h>

int main()
{
  int r = 0;
  for (int pass = 0; pass < 2; ++pass)
  {
    if (pass == 1)
      goto X;
    {
      int i = 5;
      continue;
    WAIT:
      r = i;
      break;
    }
  }
  goto END;
X:
  goto WAIT;
END:
  // A jump to X runs the goto's entry into i's block, so i is indeterminate
  assert(r == 5);
}
