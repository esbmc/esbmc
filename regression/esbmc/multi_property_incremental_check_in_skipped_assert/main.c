/* Every claim raised in the closing round's base case is promoted when the
   forward condition holds. */
#include <assert.h>

int main()
{
  int a[3] = {0, 0, 0};
  int *p = a;
  for (int i = 0; i < 3; i++)
  {
    assert(*p == 0);
    p++;
  }
  return 0;
}
