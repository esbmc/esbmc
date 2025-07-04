#include <assert.h>
#include <stdlib.h>

void unknown_method(int *a);

int main()
{
  int *p = (int *) malloc(4);

  unknown_method(p);
  free(p); // invalid pointer freed
}
