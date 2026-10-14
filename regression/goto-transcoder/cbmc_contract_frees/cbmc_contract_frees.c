#include <stdlib.h>

void release(int *p)
  __CPROVER_requires(__CPROVER_is_fresh(p, sizeof(int)))
  __CPROVER_assigns(__CPROVER_object_whole(p))
  __CPROVER_frees(p)
{
  free(p);
}

int main() { return 0; }
