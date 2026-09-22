// A component the model does not pin down -- get() reads a pointer-typed
// member back as an expression, not a value -- keeps reporting the object the
// SSA assignment rewrote, rather than a step with no value at all.
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
  __ESBMC_assert(h.t != 3, "t is 3");
  return 0;
}
