/* assert(*p == 0) fails at k = 1 and is skipped; the bounds check raised in
   its guard goes with it, so no later base case solves that check. The
   forward condition must not promote it: it fails at i == 2. */
#include <assert.h>

int main()
{
  int a[2] = {5, 5};
  int *p = a;
  for (int i = 0; i < 3; i++)
  {
    assert(*p == 0);
    p++;
  }
  return 0;
}
