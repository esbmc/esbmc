#include <assert.h>

// Failing counterpart of array_flattener_z3_struct_field: the same struct with
// an array member under --array-flattener on Z3, with a refutable assertion.
struct s
{
  int n;
  int a[4];
};

struct s g1, g2;
_Bool nondet_bool(void);
int nondet_int(void);

int main()
{
  struct s *p = nondet_bool() ? &g1 : &g2;
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 4);
  p->a[i] = i;
  assert(p->a[i] != i);
  return 0;
}
