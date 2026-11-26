/* An earlier loop's back edge sits right before the hoisted closure call. */
unsigned nondet_uint(void);
struct env
{
  unsigned *e, *acc;
};
_Bool closure(struct env *c)
{
  return *c->acc == 1;
}
int main()
{
  unsigned e = nondet_uint(), base = 3, acc = 1, k = 0;
  struct env c = {&e, &acc};
  while (k < 2)
    k++;
  while (e > 1)
    __CPROVER_loop_invariant(closure(&c) != 0 && 1)
  {
    if (e & 1)
      acc *= base;
    base *= base;
    e /= 2;
  }
  return 0;
}
