#include <stdlib.h>

int main(argc, argv) char **argv;
{
  int *foo = __builtin_alloca(sizeof(int));
  return 0;
}

