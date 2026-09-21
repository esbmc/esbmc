extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }
extern unsigned char __VERIFIER_nondet_uchar(void);

// The falsifying half of ts_kind_no_loop: a loop-free program whose property
// is violable, so the prefix check must find it rather than report success on
// an empty step.
int main() {
  unsigned char a = __VERIFIER_nondet_uchar();
  unsigned char b = __VERIFIER_nondet_uchar();
  __VERIFIER_assert((unsigned char)(a + b) != 0);
  return 0;
}
