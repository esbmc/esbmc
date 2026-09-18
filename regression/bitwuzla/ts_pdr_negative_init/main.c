// x starts negative and only ever decreases, so PDR must see the initial
// value's bits correctly to prove x < 0.
extern void abort(void);
extern _Bool __VERIFIER_nondet_bool(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

int main()
{
  signed char x = -3;
  for (;;)
  {
    __VERIFIER_assert(x < 0);
    if (__VERIFIER_nondet_bool() && x > -100)
      x = x - 1;
  }
}
