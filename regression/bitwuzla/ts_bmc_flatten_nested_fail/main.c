extern unsigned char __VERIFIER_nondet_uchar(void);
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }
int main() {
  unsigned char x = 0;
  for (;;) {
    while (__VERIFIER_nondet_uchar())
      x = x + 1;
    __VERIFIER_assert(x != 3);
  }
}
