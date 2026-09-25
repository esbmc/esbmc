// _fail sibling of github_4715_irep2_native_body_rollback_01: pins that a
// genuine violation surviving the rollback path is still reported as
// VERIFICATION FAILED, not silently dropped.
#include <assert.h>

int counter = 0;

int first_check(void)
{
  return counter < 2;
}

int main(void)
{
  int total = 0;
  while (first_check())
  {
    total = total + 1;
    counter = counter + 1;
  }

  switch (counter)
  {
  case 2:
    total = total + 10;
    break;
  default:
    total = total + 20;
    break;
  }

  assert(total == 13);
  return 0;
}
