// A boolean state bit, and an initial x that depends on it, so init is not a constant.
extern unsigned char __VERIFIER_nondet_uchar(void);
extern _Bool __VERIFIER_nondet_bool(void);
extern float __VERIFIER_nondet_float(void);
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }
int main() {
  _Bool f = __VERIFIER_nondet_bool();
  unsigned char x = f ? 2 : 4;
  for (;;) {
    __VERIFIER_assert((x & 1) == 0);
    f = !f;
    if (f)
      x = x + 2;
  }
}
