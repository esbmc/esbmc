/* The loop can run forever, but only after abort(), so every execution
 * terminates. */
#include <stdlib.h>
extern int __VERIFIER_nondet_int(void);

int main(void)
{
  int x = __VERIFIER_nondet_int();
  if (x > 0)
  {
    abort();
    while (__VERIFIER_nondet_int())
      x++;
  }
  return 0;
}
