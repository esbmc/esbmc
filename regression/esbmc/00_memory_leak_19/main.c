#include <stdlib.h>

int main()
{

  int *a = (int *) malloc(sizeof(int));

  if(a != NULL)
    return 0;

  free(a);

  return 0;
}
