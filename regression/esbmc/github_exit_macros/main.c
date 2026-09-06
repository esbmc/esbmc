/* C17 7.22: exit(EXIT_SUCCESS) must be equivalent to exit(0). The bundled
   <stdlib.h> had the two macros inverted, so a program returning EXIT_SUCCESS
   reported failure and one returning EXIT_FAILURE reported success. */
#include <assert.h>
#include <stdlib.h>

int main(void)
{
  assert(EXIT_SUCCESS == 0);
  assert(EXIT_FAILURE != 0);
  assert(EXIT_SUCCESS != EXIT_FAILURE);
  return EXIT_SUCCESS;
}
