#include <stdlib.h>

int main()
{
  char *a = malloc(200);
  __ESBMC_get_object_size(a);
}
