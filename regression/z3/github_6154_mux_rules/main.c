// GitHub #6154: differential check of the branch-merge rewrites.  Each function
// is evaluated twice over the same nondeterministic array: once outside any
// quantifier, where BMC unrolls it normally, and once under __ESBMC_exists,
// where it must be summarized (an existential cannot fall back to
// skolemization).  Any rewrite that changes the summarized value diverges from
// the unrolled reference and fails here.
//
// Every call takes the bound variable as an argument and reads a[k] in the loop
// guard.  That dependence is what forces the call through the summarizer: a
// body that does not mention the bound variable is simply hoisted to a temp and
// evaluated by ordinary BMC, which would make this test vacuous.
#define N 8
#define K 3

// Running value on the left of a non-commutative `-'.
int sub_left(int a[N], int k)
{
  int r = 100;
  for (int i = 0; i < N; ++i)
    if (a[i] > a[k])
      r = r - 3;
  return r;
}

// Running value on the *right* of `-': must not absorb as if commutative.
int sub_right(int a[N], int k)
{
  int r = 1;
  for (int i = 0; i < N; ++i)
    if (a[i] > a[k])
      r = 50 - r;
  return r;
}

// Opposite operators per branch, reconciled by the `A - k' -> `A + (-k)' rule.
int incdec(int a[N], int k)
{
  int r = 0;
  for (int i = 0; i < N; ++i)
    if (a[i] > a[k])
      ++r;
    else
      --r;
  return r;
}

// Same operator, differing constant: merged by pushing the mux into the operand.
// Spelled out rather than `+=', which the summarizer does not model.
int addk(int a[N], int k)
{
  int r = 0;
  for (int i = 0; i < N; ++i)
    if (a[i] > a[k])
      r = r + 1;
    else
      r = r + 2;
  return r;
}

// Multiplicative accumulator, whose identity is 1.  Kept shallow: mixing a
// nonlinear accumulator with the bitwise one below on a single variable costs
// Z3 two orders of magnitude more time than either alone.
int mul(int a[N], int k)
{
  int r = 1;
  for (int i = 0; i < 4; ++i)
    if (a[i] > a[k])
      r = r * 3;
  return r;
}

// Bitwise accumulator, whose identity is 0.
int xr(int a[N], int k)
{
  int r = 0;
  for (int i = 0; i < N; ++i)
    if (a[i] > a[k])
      r = r ^ 5;
  return r;
}

// Unsigned wraparound: the rewrite is exact only in modular arithmetic.
unsigned wrap(int a[N], int k)
{
  unsigned r = 0;
  for (int i = 0; i < N; ++i)
    if (a[i] > a[k])
      r = r - 7u;
  return r;
}

// Mixed-width arms.  Both arms are `int', so the mux may be pushed under the
// common typecast only if the operands themselves agree in width -- otherwise
// the wider arm is silently narrowed and the summary computes (int)(char)l.
int widths(int a[N], int k, char c, long l)
{
  int r = 0;
  for (int i = 0; i < N; ++i)
    if (a[i] > a[k])
      r = (int)c;
    else
      r = (int)l;
  return r;
}

// Equal width, differing signedness: the arms must not be unified on width
// alone, or the summary reinterprets one of them.
int signs(int a[N], int k, unsigned char uc, signed char sc)
{
  int r = 0;
  for (int i = 0; i < N; ++i)
    if (a[i] > a[k])
      r = (int)uc;
    else
      r = (int)sc;
  return r;
}

// Early return, folded by the same rewrite as the branch merge.
int early(int a[N], int k)
{
  int r = 0;
  for (int i = 0; i < N; ++i)
  {
    if (a[i] == a[k] + 42)
      return -1;
    if (a[i] > a[k])
      ++r;
  }
  return r;
}

int main()
{
  int a[N], i;
  int r1 = sub_left(a, K), r2 = sub_right(a, K), r3 = incdec(a, K);
  int r4 = addk(a, K), r5 = mul(a, K), r6 = xr(a, K), r8 = early(a, K);
  unsigned r7 = wrap(a, K);
  // 300 does not fit in a char: a narrowing mux would compute 44 instead.
  int r9 = widths(a, K, 1, 300);
  // 200 and -56 share the low 8 bits: unifying on width alone confuses them.
  int r10 = signs(a, K, 200, -56);

  __ESBMC_assert(
    __ESBMC_exists(&i, i == K && sub_left(a, i) == r1), "sub_left");
  __ESBMC_assert(
    __ESBMC_exists(&i, i == K && sub_right(a, i) == r2), "sub_right");
  __ESBMC_assert(__ESBMC_exists(&i, i == K && incdec(a, i) == r3), "incdec");
  __ESBMC_assert(__ESBMC_exists(&i, i == K && addk(a, i) == r4), "addk");
  __ESBMC_assert(__ESBMC_exists(&i, i == K && mul(a, i) == r5), "mul");
  __ESBMC_assert(__ESBMC_exists(&i, i == K && xr(a, i) == r6), "xor");
  __ESBMC_assert(__ESBMC_exists(&i, i == K && wrap(a, i) == r7), "wrap");
  __ESBMC_assert(__ESBMC_exists(&i, i == K && early(a, i) == r8), "early");
  __ESBMC_assert(
    __ESBMC_exists(&i, i == K && widths(a, i, 1, 300) == r9), "widths");
  __ESBMC_assert(
    __ESBMC_exists(&i, i == K && signs(a, i, 200, -56) == r10), "signs");
  return 0;
}
