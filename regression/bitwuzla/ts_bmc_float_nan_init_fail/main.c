// f starts as NaN, so f == f fails in the first iteration. Relating the loop
// state to its initial value with IEEE equality would admit no initial state.
extern void abort(void);
extern float __VERIFIER_nondet_float(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

int main()
{
  float f = __VERIFIER_nondet_float();
  if (f == f)
    return 0;
  for (;;)
  {
    __VERIFIER_assert(f == f);
    f = f * 1.0f;
  }
}
