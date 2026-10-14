#include <stdlib.h>

int main()
{
  int *a = malloc(sizeof(int));
  int *b = malloc(sizeof(int));

  // Relational comparison between pointers to two different heap objects
  // is undefined behaviour (C11 6.5.8p5); ESBMC should flag it with an
  // intuitive message rather than the old opaque "Same object violation".
  if (a < b)
    return 1;

  return 0;
}
