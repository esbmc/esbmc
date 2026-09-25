// A counterexample step prints an lvalue and a value; the value must be the
// one that lvalue holds. symex rewrites `s.total = ...` into a whole-object
// update of `s`, so reading the step's RHS answers for all of `s` instead.
struct sample
{
  int total;
  int reading[4];
};

struct sample s;

int main(void)
{
  for (int i = 0; i < 4; i++)
  {
    s.reading[i] = i * 3;
    s.total = s.total + s.reading[i];
  }

  __ESBMC_assert(s.total != 18, "total is 18");
  return 0;
}
