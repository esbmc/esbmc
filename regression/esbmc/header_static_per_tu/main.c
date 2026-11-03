#include <assert.h>
#include "h.h"
int from_a(void);
int main(void)
{
  int x = from_a();
  int y = counter();
  assert(x == 2 && y == 1 && g == 1);
  return 0;
}
