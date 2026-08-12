// example.c
#include <assert.h>

 int main () {
 
   /* The issue's reproducer reads uninitialised locals, which is undefined
      behaviour and is not modelled as nondeterministic on every target -- on
      LLP64/Windows the assumes below become infeasible and the assertion is
      reported UNREACHED. nondet_int() states the intent portably. */
   int x = nondet_int();
   int y = nondet_int();
   
   __ESBMC_assume (x < 100);
   __ESBMC_assume (x > 0);
   
   __ESBMC_assume (y < 100);
   __ESBMC_assume (y > 0);
   
   int c = x + y;
  
  __ESBMC_assume (c < 100);
  assert(c < 100); 
  
  return c;
}
