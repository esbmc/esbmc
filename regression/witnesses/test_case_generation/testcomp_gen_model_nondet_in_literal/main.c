#include <assert.h>
#include <time.h>

extern int __VERIFIER_nondet_int(void);

struct timespec ts;

int main()
{
  /* timespec_get's model sets ts.tv_nsec from a nondet of its own, which
   * then sits in the literal `ts == { .tv_sec=5, .tv_nsec=nondet1 }`. It is
   * not a harness input: the only one is d. */
  timespec_get(&ts, TIME_UTC);
  ts.tv_sec = 5;
  int d = __VERIFIER_nondet_int();
  if (d == 22)
    assert(0);
  return 0;
}
