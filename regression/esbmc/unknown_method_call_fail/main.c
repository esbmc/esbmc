#include <assert.h>
#include <stdlib.h>

void unknown_method(int *a);

int main()
{
  int *p = (int *) malloc(4);
  free(p);

  unknown_method(p);
}
