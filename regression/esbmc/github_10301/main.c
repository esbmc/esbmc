// #10301: realloc of a pointer malloc returned is valid.
#include <stdlib.h>
int main()
{
  int *q = malloc(sizeof(int));
  int *r = realloc(q, 2 * sizeof(int));
  free(r ? r : q);
}
