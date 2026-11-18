// The assertion before the loop fails for x == 3.
extern unsigned char __VERIFIER_nondet_uchar(void);
extern _Bool __VERIFIER_nondet_bool(void);
extern float __VERIFIER_nondet_float(void);
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }
int main() {
  unsigned char x = __VERIFIER_nondet_uchar();
  __VERIFIER_assert(x != 3);
  for (;;)
    x = x + 1;
}
