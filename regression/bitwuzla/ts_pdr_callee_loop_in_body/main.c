// g's loop gets the k-induction havoc followed by an unconditional assume of
// its condition, false for g(0); left in, PDR took that assume as an invariant
// and proved the program safe. Extraction must refuse it.
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }
int K = 1;
unsigned g(unsigned n)
{
  unsigned i = 0;
  while (i < n)
  {
    if (K)
      return 1;
    i++;
  }
  return 0;
}
int main()
{
  unsigned char x = 0;
  while (x < 10)
  {
    unsigned r = g(x);
    __VERIFIER_assert(r <= 1);
    x++;
  }
  return 0;
}
