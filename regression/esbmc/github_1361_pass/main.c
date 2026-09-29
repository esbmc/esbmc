/* Safe twin of github_1361: one table, two passing rows, and the base cases
   report no verdict of their own. */
#include <assert.h>

int main()
{
  for (int i = 0; i < 2; i++)
  {
    assert(i < 5);
  }
  assert(1 != 2);
  return 0;
}
