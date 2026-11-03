/* github #8703: strtol sets errno to ERANGE on overflow. */
#include <assert.h>
#include <errno.h>
#include <stdlib.h>
int main(void)
{
  errno = 0;
  strtol("99999999999999999999", NULL, 10);
  assert(errno == 0);
  return 0;
}
