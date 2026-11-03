// As pointer_write_heap_fresh_kept_fail, with an allocation on every
// iteration of which p keeps only the first.
#include <stdlib.h>
#include <assert.h>
extern int nondet_int(void);
int main()
{
  int *p = 0;
  while (nondet_int())
  {
    int *t = malloc(sizeof(int));
    *t = 0;
    if (!p)
      p = t;
    *p = *p + 1;
    assert(*p < 3);
  }
  return 0;
}
