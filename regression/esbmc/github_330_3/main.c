#include <stdlib.h>

int main(argc, argv) char **argv;
{
  int *foo = __builtin_alloca(sizeof(int));
  free(foo);
  return 0;
}

