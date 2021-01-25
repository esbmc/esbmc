#include <stdlib.h>

struct S
{
  int x;
  char a[];
};

int main()
{
  struct S *p=malloc(sizeof(struct S)+10);
  __ESBMC_assume(p);
  
  p->x=1;
  p->a[0]=3;
  p->a[9]=3;
}
