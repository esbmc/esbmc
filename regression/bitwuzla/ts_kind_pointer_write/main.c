// The loop writes x only through p. Havocking x in the inductive step still
// proves the property, since (x + 1) % 10 < 10 for every int x.
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

int main()
{
  int x = 0;
  int *p = &x;
  for (;;)
  {
    *p = (*p + 1) % 10;
    __VERIFIER_assert(x < 10);
  }
}
