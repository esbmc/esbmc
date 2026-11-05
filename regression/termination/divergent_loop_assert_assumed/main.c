/* The loop can run forever, but only after the assertion has failed, which
 * aborts: no execution reaches it. --termination --no-assertions skips the
 * assertion without ending the path, so termination is not proved. */
#include <assert.h>
extern int __VERIFIER_nondet_int(void);

int main(void)
{
  int x = __VERIFIER_nondet_int();
  assert(x > 0);
  if (x <= 0)
    while (__VERIFIER_nondet_int())
      ;
  return 0;
}
