// y is declared inside the loop without an initialiser, so each iteration
// may see a different value. The violation needs y == 0 in one iteration and
// y == 1 in the next.
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

int main()
{
  int prev = -1;
  for (;;)
  {
    int y;
    __VERIFIER_assert(!(prev == 0 && y == 1));
    prev = y;
  }
}
