// The falsifying half of ts_pdr_bool_symbolic_init: x becomes odd.
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
      x = x + 1;
  }
}
