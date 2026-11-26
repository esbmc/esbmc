// The assertion after the loop is only reached by leaving it: the step that
// exits runs the suffix, so the transition system itself finds the bug.
extern unsigned char __VERIFIER_nondet_uchar(void);
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }
int main() {
  unsigned char x = 0;
  while (__VERIFIER_nondet_uchar())
    x = x + 1;
  __VERIFIER_assert(x != 2);
  return 0;
}
