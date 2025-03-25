#include <stdlib.h>
#include <assert.h>

int *a;
int n;

int test(){

   int i;

   for (i = 0; i < n; ++i){
     if (a[i])
       n--;
      
     a[i] = 0;            
   }

   return 0;
}

int main(){

   n = nondet_int();
   
   if (n <= 0 || n >= 10){
      n=5;
      a = (int *) malloc(n * sizeof(int));
      __ESBMC_assume(a);
   } else {      
      assert(n>0 && n<10);
      a = (int *) malloc( n * sizeof(int));
      __ESBMC_assume(a);
   }

   test();

   return 1;
}
