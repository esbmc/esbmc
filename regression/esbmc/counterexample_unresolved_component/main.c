// The holding counterpart of counterexample_unresolved_component_fail.
struct holder
{
  int *p;
  int t;
};

struct holder h;
int x = 5;

int main(void)
{
  h.p = &x;
  h.t = 3;
  __ESBMC_assert(h.t == 3, "t is not 3");
  return 0;
}
