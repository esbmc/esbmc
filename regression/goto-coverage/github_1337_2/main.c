#include <assert.h>

int main()
{
   int x = 0;
   while (nondet_int())
   {
      if (!x)
      {
         assert(x == 0);
         x = 1;
      }
      else if (x == 1)
      {
         assert(x > 0);
         x = 2;
      }
      else if (x == 2)
      {
         assert(x >= 2);
         x = 3;
      }
   }
   assert(x == 3);
}