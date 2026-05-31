// Companion to regression #4986. Same void*/char* pointer-difference encoding
// path inside the memmove operational model that used to abort() in
// smt_convt::convert_pointer_arith, but here a genuine signed overflow sits
// downstream of the copied byte. The fix must let the pointer difference be
// encoded *and* must not mask the real violation: ESBMC has to still report
// VERIFICATION FAILED (arithmetic overflow on the multiplication).
#include <string.h>
#include <stdlib.h>

extern unsigned __VERIFIER_nondet_uint(void);

int main(void)
{
  int used = (int)__VERIFIER_nondet_uint();
  __ESBMC_assume(used >= 1 && used <= 2);
  int applet_len = (int)__VERIFIER_nondet_uint();
  __ESBMC_assume(applet_len >= 1 && applet_len <= 2);

  char *msg = (char *)malloc((unsigned long)(used + 1));
  if (msg == (char *)0)
    return 0;
  char *msg1 = (char *)realloc((void *)msg, (unsigned long)(applet_len + used + 3));
  if (msg1 == (char *)0)
    return 0;
  msg = msg1;

  memmove((void *)(msg + (long)applet_len), (const void *)msg, (unsigned long)used);

  // The copied byte keeps the void*/char* pointer-difference live, then feeds
  // a multiplication that overflows for any |x| >= 2 -> CWE-190 must fire.
  int x = (int)msg[(long)applet_len];
  int y = x * 2147483647;
  return y;
}
