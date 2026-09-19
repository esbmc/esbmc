// inc writes x through its parameter, resolved in the callee's scope.
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

int x;
void inc(int *q) { *q = *q + 1; }

int main()
{
  int i = 0;
  for (;;)
  {
    inc(&x);
    i++;
    __VERIFIER_assert(x < 10);
  }
}
