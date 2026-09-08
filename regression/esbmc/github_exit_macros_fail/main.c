/* EXIT_SUCCESS is 0, so this does not hold. */
#include <assert.h>
#include <stdlib.h>

int main(void)
{
  assert(EXIT_SUCCESS == 1);
  return 0;
}
