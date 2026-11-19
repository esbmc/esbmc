// PDR works on bit-vector cubes, so float state ends UNKNOWN rather than in a verdict.
extern unsigned char __VERIFIER_nondet_uchar(void);
extern _Bool __VERIFIER_nondet_bool(void);
extern float __VERIFIER_nondet_float(void);
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }
int main() {
  float f = 0.0f;
  for (;;) {
    __VERIFIER_assert(f >= 0.0f);
    f = f + 1.0f;
  }
}
