// The loop is entered by a conditional jump to its test, which gets no havoc
// (see github_7565_conditional_entry_fail), so nothing in it may count as
// reassigned and the inductive step stays off.
#include <assert.h>
#include <stdlib.h>
extern int nondet_int(void);
int main()
{
  int *buf = malloc(sizeof(int) * 2);
  int *p;
  int i = 0, x = 0;
  if (nondet_int())
    goto test;
  return 0;
body:
  p = buf + 1;
  *p = i;
  i++;
  if (i == 3)
    x = 1;
test:
  assert(x == 0);
  if (i < 6)
    goto body;
  return 0;
}
