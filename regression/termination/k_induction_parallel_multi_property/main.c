#include <stdbool.h>

// Regression for issue #6031: combining --termination with
// --k-induction-parallel produced non-deterministic, sometimes unsound
// VERIFICATION SUCCESSFUL. The program does not terminate (the while(true)
// loop never exits), so --termination must report VERIFICATION FAILED
// deterministically. --termination now takes priority over the parallel
// k-induction driver, which has no termination-property interpretation.
int a;
int main()
{
  while (true)
  {
    if (a == 0)
      a = 1;
    else if (a == 1)
      a = 2;
    else if (a == 2)
    {
      if (true)
        a = 3;
    }
    else
      __VERIFIER_error();
  }
  return 0;
}
