#include <setjmp.h>

extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

jmp_buf env;
int global = 0;

// src/c2goto/library/setjmp.c models setjmp and longjmp as
// __ESBMC_unreachable(), which means "not modelled" but becomes a
// reachable-error property under --enable-unreachability-intrinsic. Answering
// here would report a violation that is not in the program: `global` is never
// written, so the assertion holds. Extraction must decline instead.
int main() {
  __VERIFIER_assert(global == 0);
  setjmp(env);
  longjmp(env, 2);
  return 0;
}
