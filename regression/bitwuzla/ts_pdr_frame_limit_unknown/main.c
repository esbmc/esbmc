// Safe, but needs more frames than --max-k-step allows.
extern unsigned char __VERIFIER_nondet_uchar(void);
extern _Bool __VERIFIER_nondet_bool(void);
extern float __VERIFIER_nondet_float(void);
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }
int main() {
  unsigned char x = 0;
  for (;;) {
    __VERIFIER_assert(x != 12);
    if (__VERIFIER_nondet_bool() && x < 10)
      x = x + 2;
    else
      x = x != 0 ? x - 1 : 0;
    x = x & 15;
  }
}
