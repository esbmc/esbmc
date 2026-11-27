/* Kani lowers a loop invariant to a call of a generated closure; k-induction
 * havocs every variable the loop writes ahead of the loop head. */
unsigned nondet_uint(void);
struct env
{
  unsigned *i, *n;
};
_Bool closure(struct env *c, int k)
{
  return *c->i <= *c->n && *c->i < 1000 && k == 1;
}
int main()
{
  unsigned n = nondet_uint(), i = 0;
  unsigned a = 0, b = 0, d = 0, e = 0, f = 0, g = 0, h = 0, j = 0, k = 0;
  struct env c = {&i, &n};
  while (i < n)
    __CPROVER_loop_invariant(closure(&c, 1) != 0 && 1)
    {
      ++i;
      ++a, ++b, ++d, ++e, ++f, ++g, ++h, ++j, ++k;
    }
  __CPROVER_assert(i == n, "i reaches n");
  return 0;
}
