/* An expression statement whose value has array type -- `y->ss;` where y points
   at a struct with an array member -- is rewritten to `&y->ss[0]`, because the
   dereference code does not assume such an object exists. The statement's value
   is unused, so taking the first element's address is free. An assignment
   operand is exempt: there the array is the target, not a discarded value. */
#include <assert.h>

struct Base
{
  int ss[128];
};

int arr[4];

int main(void)
{
  struct Base x, *y = &x;

  x.ss[0] = 5;
  y->ss;  /* the shape that needed the rewrite */
  arr;    /* a plain array-typed statement too */

  assert(x.ss[0] == 5);
  return 0;
}
