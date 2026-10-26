/* An array of structs holding a vector, written at a symbolic index. */
#include <assert.h>
typedef int v4i __attribute__((vector_size(16)));
struct S
{
  v4i v;
  int k;
};
int nondet_int();
int main()
{
  struct S a[2];
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);
  a[i].v = (v4i){1, 2, 3, 4};
  a[i].k = 5;
  assert(a[i].v[2] == 3 && a[i].k == 5);
  return 0;
}
