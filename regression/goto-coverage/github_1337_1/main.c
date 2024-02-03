#include <assert.h>

int main()
{

   int x[2];
   x[3] = 10;
   switch (nondet_int())
   {
   case 1:
      assert(0);
   case 2:
      assert(0);
   default:
       ;
   }
   return 0;
}