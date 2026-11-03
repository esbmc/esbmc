// As pointer_write_heap_root_truncated_fail, with the truncated address in a
// global another function assigns.
#include <stdlib.h>
#include <assert.h>
extern int nondet_int(void);
unsigned G;
int *P;
void set(void) { G = (unsigned)(unsigned long)P; }
int main()
{
  P = malloc(sizeof(int));
  *P = 0;
  set();
  int *r;
  while (nondet_int())
  {
    r = (int *)(unsigned long)G;
    *r = *r + 1;
    assert(*P < 3);
  }
  return 0;
}
