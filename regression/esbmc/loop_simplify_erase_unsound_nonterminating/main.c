/* Soundness: Path 1 must not erase a loop whose body writes to a
 * dying variable but whose exit condition is never satisfied. The
 * loop below never exits (i stays 0), so the assertion is
 * unreachable. A pre-fix Path 1 would erase the loop entirely, let
 * control fall through, and report a spurious VERIFICATION FAILED. */
#include <assert.h>
int main()
{
  {
    int i = 0;
    while (i < 10)
      i = i; /* no-op: keeps i unchanged, loop never exits */
  }
  assert(0); /* unreachable in reality */
  return 0;
}
