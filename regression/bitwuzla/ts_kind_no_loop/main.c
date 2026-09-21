extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }
extern unsigned char __VERIFIER_nondet_uchar(void);

// No loop: the whole program is the transition system's prefix, so there is no
// state and no step, and check_prefix alone decides it.
int main() {
  unsigned char a = __VERIFIER_nondet_uchar();
  unsigned char b = __VERIFIER_nondet_uchar();
  __VERIFIER_assert((unsigned char)(a + b) == (unsigned char)(b + a));
  return 0;
}
