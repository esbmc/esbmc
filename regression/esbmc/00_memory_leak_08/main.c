#include <stdlib.h>

int main(){

   int * a;
   int i,j;
   int k = 2;//nondet_int(); //__NONDET__();

   if ( k <= 0 ) return -1;
   
   a= malloc( k * sizeof(int));
   __ESBMC_assume(a);

//   __ESBMC_assume(a);
//   __ESBMC_assume(k >= 100);
   
   for (i =0 ; i != k; i++)
      if (a[i] <= 1) break;

   i--;
   
   for (j = 0; j < i; ++j)
      a[j] = a[i];
   

   return 0;

}
