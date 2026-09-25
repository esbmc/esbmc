#include <assert.h>
#include <string.h>
int main()
{
  char a[5] = {1, 2, 3, 4, 5};
  memmove(a + 1, a, 4); // overlapping copy: {1,2,3,4} shifted into a[1..4]
  assert(a[1] == 1 && a[2] == 2 && a[3] == 3 && a[4] == 4);
  return 0;
}
