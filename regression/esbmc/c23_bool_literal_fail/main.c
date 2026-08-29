#include <assert.h>

/* C23 6.3.1.2 still normalises to 0 or 1, so this must be reported. */
int main(void)
{
  bool n = (bool)7;
  assert(n != true);

  return 0;
}
