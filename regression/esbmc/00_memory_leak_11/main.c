//#include <stdio.h>
#include <stdlib.h>

void really ( void );

int main ( void )
{ 
  char *memoryArea = malloc(10);
  __ESBMC_assume(memoryArea);
  char *newArea = malloc(10);
  __ESBMC_assume(newArea);

  memoryArea[3]= newArea;

  free(memoryArea);

  return 0;
}
