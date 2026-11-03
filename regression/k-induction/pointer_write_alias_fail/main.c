// The loop moves p through pp, so havocking *p at the loop head misses b.
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

int main()
{
  int a = 0, b = 0;
  int *p = &a;
  int *r = &b;
  int **pp = &p;
  int i = 0;
  for (;;)
  {
    *pp = r;
    *p = *p + 1;
    i++;
    __VERIFIER_assert(b < 10);
  }
}
