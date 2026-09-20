// p starts uninitialised, which must not make it point anywhere: the loop
// writes x through it, and havocking x still proves the bound.
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

int x;

int main()
{
  int *p;
  for (;;)
  {
    p = &x;
    *p = (*p + 1) % 10;
    __VERIFIER_assert(x < 10);
  }
}
