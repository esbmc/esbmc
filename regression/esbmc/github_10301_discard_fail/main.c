// #10301: a discarded realloc result crashed GOTO conversion.
#include <stdlib.h>
int main()
{
  int *q = (int *)1;
  realloc(q, sizeof(int));
}
