extern unsigned char __VERIFIER_nondet_uchar(void);
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }
void assume_abort_if_not(int c) { if (!c) abort(); }
int main() {
  for (;;) {
    unsigned char in = __VERIFIER_nondet_uchar();
    assume_abort_if_not(in != 3);
    __VERIFIER_assert(in != 3);
  }
}
