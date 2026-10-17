#include <assert.h>
int main(void)
{
  int m = 3, a[m];
  assert(sizeof(a) == 2 * sizeof(int));
  return 0;
}
