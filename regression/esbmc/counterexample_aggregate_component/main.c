// The holding counterpart of counterexample_aggregate_component_fail.
union u
{
  int i;
  char c[4];
};

struct holder
{
  union u un;
  int t;
};

struct holder harr[2];

int main(void)
{
  struct holder v;
  v.un.i = 9;
  v.t = 4;
  harr[0] = v;
  __ESBMC_assert(harr[0].t == 4, "t is not 4");
  return 0;
}
