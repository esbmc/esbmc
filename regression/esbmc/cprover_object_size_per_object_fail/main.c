#include <assert.h>
#include <stddef.h>

/* The smaller of the two objects a pointer may address keeps its own size. */
_Bool nondet_bool(void);

int main(void)
{
  char small[1];
  char big[100];
  char *p = nondet_bool() ? small : big + 50;
  assert(__CPROVER_OBJECT_SIZE(p) == 100);
  return 0;
}
