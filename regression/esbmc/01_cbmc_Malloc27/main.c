//#include <stdio.h>
#include <stdlib.h>

void really ( char* p );

int main ( void )
{ 
   char* p = malloc(10);
   really(p);
   return 0;
}

void really ( char* p )
{
   int i;
   for (i = 0; i < 1; i++)
      p[i] = 'z';
   free(p);
   p[1] = 'z';
   p = malloc(10);
   p[2] = 'z';
}
