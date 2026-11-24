#include <assert.h>

// A struct with an array member, written through a pointer, under
// --array-flattener on Z3. Z3's native tuples cannot hold a flattened array as
// a field, so the flattener must also flatten tuples; before that, building the
// struct passed an array_convt ast to z3_convt::tuple_create.
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
  assert(p->a[i] == i);
  return 0;
}
