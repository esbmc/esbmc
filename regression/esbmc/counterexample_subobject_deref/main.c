// The holding counterpart of counterexample_subobject_deref_fail.
struct inner
{
  int pad;
  int g;
};

struct outer
{
  int g;
  struct inner in;
};

struct outer o;

static void set(struct inner *p)
{
  p->g = 5;
}

int main(void)
{
  o.g = 9;
  set(&o.in);
  __ESBMC_assert(o.in.g == 5, "in.g is 5");
  return 0;
}
