// GitHub #6154: `while' and `do-while' loops (not just `for') with a
// statically constant trip count, and a `--'/`++' arm on both sides, inside a
// quantifier body.  sumn unrolls a `while'; inc_once unrolls a `do-while' that
// executes its body once before the first condition check.
int sumn(int x)
{
  int i = 0;
  int r = x;
  while (i < 3)
  {
    r = r + 1;
    i = i + 1;
  }
  return r;
}

int dec_twice(int x)
{
  int r = x;
  int i = 0;
  while (i < 2)
  {
    --r;
    ++i;
  }
  return r;
}

int inc_once(int x)
{
  int r = x;
  int i = 0;
  do
  {
    r = r + 1;
    i = i + 1;
  } while (i < 1);
  return r;
}

int main()
{
  int v;
  __ESBMC_assert(__ESBMC_forall(&v, sumn(v) == v + 3), "while-loop summarized");
  __ESBMC_assert(
    __ESBMC_forall(&v, dec_twice(v) == v - 2), "while-loop with predecrement");
  __ESBMC_assert(
    __ESBMC_forall(&v, inc_once(v) == v + 1), "do-while summarized");
  return 0;
}
