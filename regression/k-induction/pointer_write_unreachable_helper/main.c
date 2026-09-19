// clear() writes through a pointer nothing resolves, but nothing calls it.
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

void clear(int *p, int n)
{
  for (int i = 0; i < n; i++)
    p[i] = 0;
}

int main()
{
  int i = 0;
  for (;;)
  {
    i = (i + 1) % 5;
    __VERIFIER_assert(i < 5);
  }
}
