#include <assert.h>
int buf[2][2] = {{1, 2}, {3, 4}};
int main(void)
{
  for (int k = 0; k < 2; k++)
  {
    int m = 2 - k, (*p)[m] = (void *)buf;
    if (k == 1)
      assert(p[1][0] == 3); /* m == 1: p[1][0] is buf[0][1] == 2 */
  }
  return 0;
}
