#include <assert.h>
#include "h.h"
int from_a(void);
int main(void)
{
  int x = from_a();
  int y = counter();
  assert(y == 3);
  return 0;
}
