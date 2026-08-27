// Control: the same loop with a bound that covers it. Nothing is truncated,
// so the warning must not appear.
#include <assert.h>

int main(void)
{
  int s = 0;
  for (int i = 0; i < 10; i++)
    s++;
  assert(s == 10);
  return 0;
}
