// example.c
#include <assert.h>

 int main () {
 
   int x ;
   int y ;
   
   __ESBMC_assume (x < 100);
   __ESBMC_assume (x > 0);
   
   __ESBMC_assume (y < 100);
   __ESBMC_assume (y > 0);
   
   int c = x + y;
  
  __ESBMC_assume (c < 100);
  assert(c < 100); 
  
  return c;
}
