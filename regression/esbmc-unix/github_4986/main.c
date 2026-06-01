// Regression for #4986: smt_convt::convert_pointer_arith used to abort()
// (SIGABRT, "Assertion failed: side1->type == side2->type") while encoding a
// pointer-difference VCC whose two operands had structurally different but
// same-sized pointer subtypes.
//
// The pattern is the one ESBMC's <string.h> memmove operational model
// performs internally: `if (dest - src >= n)`. Both parameters are declared
// void*, but the value-set / dereference machinery specialises the `src`
// operand to the concrete element type of the object it points into (here the
// char buffer returned by realloc, subtype `unsigned char`), while `dest`
// stays a symbolic void* (subtype `empty`). C11 6.5.6p9 only requires the two
// operands to point into the same object, hence to have equal-*sized* element
// types -- which holds here (sizeof(void)==1 per the GCC extension, equal to
// sizeof(char)) -- so the encoding must succeed instead of asserting on
// identical pointer types.
//
// Distilled from the busybox no-overflow benchmarks sync-1/sync-2/
// whoami-incomplete-1/whoami-incomplete-2; this crash previously masked local
// reproduction of #4976-#4979 on aarch64. The program is safe, so the expected
// verdict is VERIFICATION SUCCESSFUL.
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

  // dest = (void*)&realloc_array[applet_len] (symbolic void*),
  // src  = (void*)&realloc_array[0]          (resolved char*).
  memmove((void *)(msg + (long)applet_len), (const void *)msg, (unsigned long)used);

  // Read a copied byte so the pointer-difference inside memmove is not sliced
  // away (it now feeds the overflow check below).
  int x = (int)msg[(long)applet_len];
  int y = x + used;
  return y;
}
