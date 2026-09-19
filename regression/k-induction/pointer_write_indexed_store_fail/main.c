// q moves, so the inductive step havocs what the points-to analysis says p0
// may hold. Missing the store pp[0] = &x would leave x at its pre-loop value.
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

int x, y;

int main()
{
  int *p0 = &y;
  int **pp = &p0;
  pp[0] = &x;
  x = 0;
  y = 0;
  for (;;)
  {
    int *q = p0;
    *q = *q + 1;
    __VERIFIER_assert(x < 10);
  }
}
