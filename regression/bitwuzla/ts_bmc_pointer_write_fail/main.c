// The loop writes x only through p. Unless the inductive step havocs what p
// points to, x keeps its pre-loop value 0 and the step proves x < 10.
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

int main()
{
  int x = 0, i = 0;
  int *p = &x;
  for (;;)
  {
    *p = *p + 1;
    i++;
    __VERIFIER_assert(x < 10);
  }
}
