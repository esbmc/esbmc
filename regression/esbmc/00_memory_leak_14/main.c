//#include <stdio.h>
#include <stdlib.h>

void really ( char* p );

int main ( void )
{ 
   char* p = malloc(10);
   __ESBMC_assume(p);
   really(p);
   return 0;
}

void really ( char* p )
{
   int i;
   for (i = 0; i < 1; i++)
      p[i] = 'z';
   free(p);
   p = malloc(10);
   __ESBMC_assume(p);
   p[1] = 'z';
   p[2] = 'z';
}
