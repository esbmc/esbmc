// _fail sibling of github_4715_irep2_native_body_while_call_01: pins that a
// genuine violation through a native side-effecting while-condition is still
// reported as VERIFICATION FAILED, not silently dropped.
#include <assert.h>

int counter = 0;

int has_more(void)
{
  return counter < 3;
}

int main(void)
{
  int total = 0;
  while (has_more())
  {
    total = total + 1;
    counter = counter + 1;
  }
  assert(total == 4);
  return 0;
}
