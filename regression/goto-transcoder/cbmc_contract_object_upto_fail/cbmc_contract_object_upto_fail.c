#include <stdlib.h>

void fill2(int *a)
  __CPROVER_requires(__CPROVER_is_fresh(a, 4 * sizeof(int)))
  __CPROVER_assigns(__CPROVER_object_upto(a, 2 * sizeof(int)))
  __CPROVER_ensures(a[0] == 7)
{
  a[0] = 7;
  a[2] = 8; /* outside the upto range */
}

int main() { return 0; }
