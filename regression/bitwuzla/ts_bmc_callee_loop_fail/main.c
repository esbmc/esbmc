// A loop in a callee that runs several times per step cannot be captured in a
// one-iteration symbolic run: extraction must refuse and the normal strategy
// must find the bug.
extern unsigned char __VERIFIER_nondet_uchar(void);
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }
unsigned char add(unsigned char x, unsigned char n) {
  for (unsigned char i = 0; i < n; i++)
    x = x + 1;
  return x;
}
int main() {
  unsigned char x = 0;
  for (;;) {
    unsigned char n = __VERIFIER_nondet_uchar() & 3;
    __VERIFIER_assert(x != 5);
    x = add(x, n);
  }
}
