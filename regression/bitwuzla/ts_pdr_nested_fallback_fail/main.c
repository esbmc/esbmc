// Not a transition system, as the flattened loop's state holds a pointer:
// --ts-pdr falls back to BMC without the inductive step.
extern unsigned char __VERIFIER_nondet_uchar(void);
extern _Bool __VERIFIER_nondet_bool(void);
extern float __VERIFIER_nondet_float(void);
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }
int main() {
  unsigned char x = 0;
  for (;;) {
    unsigned char y = x;
    unsigned char *p = &y;
    while (__VERIFIER_nondet_uchar())
      x = x + 1;
    __VERIFIER_assert(*p != 3);
  }
}
