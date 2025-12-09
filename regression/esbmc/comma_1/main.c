#include <stdio.h>
#include <assert.h>

int main() {
   int i, j;
   
   j = 10;
   i = (j++, j+100, 999+j);

   assert(i = 1010);
   
   return 0;
}
