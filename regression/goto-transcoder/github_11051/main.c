/* Kani lowers a loop invariant to a call of a generated closure. */
unsigned nondet_uint(void);
struct env
{
  unsigned *e, *exp;
};
_Bool closure(struct env *c, int k)
{
  return *c->e <= *c->exp && k == 1;
}
int main()
{
  unsigned exp = nondet_uint(), e = exp, base = 3, acc = 1;
  struct env c = {&e, &exp};
  while (e > 1)
    __CPROVER_loop_invariant(closure(&c, 1) != 0 && 1)
  {
    if (e & 1)
      acc *= base;
    base *= base;
    e /= 2;
  }
  __CPROVER_assert(e <= exp, "e never grows");
  return 0;
}
