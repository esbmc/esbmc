// on is fixed before the loop, so PDR must keep it in every cube as a state bit.
extern unsigned char __VERIFIER_nondet_uchar(void);
extern _Bool __VERIFIER_nondet_bool(void);
extern float __VERIFIER_nondet_float(void);
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }
int main() {
  _Bool on = __VERIFIER_nondet_bool();
  unsigned char x = 0;
  for (;;) {
    __VERIFIER_assert(on || x == 0);
    if (on)
      x = x + 1;
  }
}
