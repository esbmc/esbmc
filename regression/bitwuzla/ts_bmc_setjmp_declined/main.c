#include <setjmp.h>

extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

jmp_buf env;
int global = 0;

// Extraction declines any setjmp call (see calls_are_supported), even in a
// program without longjmp, where lower_longjmp leaves it alone.
int main() {
  __VERIFIER_assert(global == 0);
  setjmp(env);
  return 0;
}
