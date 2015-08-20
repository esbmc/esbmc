#include <stdlib.h>

void really ( void );

int main ( void )
{ 
   really();
   return 0;
}

void really ( void )
{
   int i;
   char* p = malloc(10);
   __ESBMC_assume(p);
   for (i = 0; i < 1; i++)
      p[i] = 'z';
   free(p);
//   p[1] = 'z';
   p = malloc(10);
   __ESBMC_assume(p);
//   p[2] = 'z';
}
