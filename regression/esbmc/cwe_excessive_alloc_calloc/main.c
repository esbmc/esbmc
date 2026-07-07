#include <stdlib.h>

extern unsigned __VERIFIER_nondet_uint(void);

// calloc has no inline sideeffect lowering; it is a real function whose
// operational model (src/c2goto/library/stdlib.c) allocates via malloc. The
// excessive-size pass instruments that model body, so an attacker-controlled
// calloc size is still flagged CWE-789. The finding is reported at the model's
// malloc site rather than this call site (documented limitation), so only the
// message and CWE tag are anchored here, not the location.
int main(void)
{
  unsigned n = __VERIFIER_nondet_uint();
  char *p = calloc(n, 1);
  free(p);
  return 0;
}
