#include <stdlib.h>
#include <assert.h>
int main(){

   int * a;
//   int k = __NONDET__();
   int k = 8;
   int i;
   if ( k <= 0 ) return -1;
   
 a = malloc( (k-1) * sizeof(int));
//   __ESBMC_assume(a != 0);
   assert(a != 0);
 //  assert(a == 0);
   for (i =0 ; i < k; i++){
      assert(a+i != 0);
      if (a[i]) return 1;
  }
   return 0;

}
