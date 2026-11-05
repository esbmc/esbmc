/* The failing assertion aborts before the loop, so every execution
 * terminates, although --no-assertions does not check the assertion. */
#include <assert.h>
extern int __VERIFIER_nondet_int(void);

int main(void)
{
  assert(0);
  while (__VERIFIER_nondet_int())
    ;
  return 0;
}
