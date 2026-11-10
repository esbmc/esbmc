/* github #10174: the do-while's else-branch must stay on its own increment,
 * not jump to the havocs of the loop that follows. */
#include <assert.h>
int main()
{
  int i = 0, a = 0, j = 0, b = 0;
  do
  {
    if (i % 2)
      a++;
    else
      b++;
    i++;
  } while (i < 4);
  while (j < 3)
  {
    j++;
    b = b + 1;
  }
  assert(b == 5 && a == 2 && j == 3);
  return 0;
}
