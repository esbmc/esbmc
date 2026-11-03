#include <assert.h>
int main(void)
{
  int n = 3;
  typedef int row[n];
  n = 5;
  row x;
  assert(sizeof(x) == 5 * sizeof(int));
  return 0;
}
