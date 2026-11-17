// PDR builds cubes from bit extracts, which integer encoding has no sort for.
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
