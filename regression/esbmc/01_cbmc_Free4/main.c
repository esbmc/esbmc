#include <stdlib.h>

void free(void *p);

void my_free(int *q)
{
  free(q);
}

int main()
{
  int *p=malloc(sizeof(int));
  __ESBMC_assume(p);
  
  *p=2;
  free(p);
//  my_free(p);

//  *p=3;
}
