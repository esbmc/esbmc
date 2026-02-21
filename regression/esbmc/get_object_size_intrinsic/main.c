#include <stdlib.h>
#include <assert.h>

int main()
{
  char *a = malloc(200);
  assert(__ESBMC_get_object_size(a) == 200);
}
