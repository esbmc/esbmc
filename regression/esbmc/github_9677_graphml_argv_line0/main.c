/* The argv setup ESBMC adds for this main has no source location. */
#include <assert.h>
void reach_error(void) { assert(0); }
extern int __VERIFIER_nondet_int(void);
int main(int argc, char *argv[])
{
  int x = __VERIFIER_nondet_int();
  if (x == 3)
    reach_error();
  return 0;
}
