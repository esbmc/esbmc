#include <stdlib.h>

int * a;

int test(int * n ){
   int i;
   for (i = 0; i < (*n); ++i){
      a[i] = 0;            
   }

   return 1;
}


int main(){

   int n = 100;//nondet_int(); //__NONDET__();
   
   if (n <= 0 || n >= 1024){
      exit(1);
   } else {
      a = (int *) malloc( n * sizeof(int));
      __ESBMC_assume(a);
   }

   //__ESBMC_assume(a);

   test(&n);

   return 1;
}
