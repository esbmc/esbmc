// #10301: realloc of a pointer no allocator returned is undefined behaviour.
#include <stdlib.h>
int main()
{
  int *q = (int *)1;
  q = realloc(q, sizeof(int));
}
