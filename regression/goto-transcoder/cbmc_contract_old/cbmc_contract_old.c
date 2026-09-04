#include <stdlib.h>

void inc(int *p)
  __CPROVER_requires(__CPROVER_is_fresh(p, sizeof(int)))
  __CPROVER_requires(*p >= 0 && *p < 1000)
  __CPROVER_assigns(*p)
  __CPROVER_ensures(*p == __CPROVER_old(*p) + 1)
{
  *p = *p + 1;
}

int main() { return 0; }
