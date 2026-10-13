// As pointer_write_heap_root_int_offset_fail, through a pointer the loop
// reassigns before writing through it.
#include <stdlib.h>
#include <assert.h>
extern int nondet_int(void);
int main()
{
  int *A = malloc(sizeof(int));
  int *B = malloc(sizeof(int));
  *A = 0;
  *B = 0;
  long off = (long)B - (long)A;
  int *r;
  while (nondet_int())
  {
    r = (int *)((long)A + off);
    *r = *r + 1;
    assert(*B < 3);
  }
  return 0;
}
