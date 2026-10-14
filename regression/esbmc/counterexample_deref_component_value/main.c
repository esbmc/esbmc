// The holding counterpart of counterexample_deref_component_value_fail.
struct pair
{
  int in;
  int out;
};

struct pair p;

static void scale(struct pair *q)
{
  q->out = q->in * 3;
}

int main(void)
{
  p.in = 7;
  scale(&p);
  __ESBMC_assert(p.out == 21, "out is not 21");
  return 0;
}
