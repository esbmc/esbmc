/* Kani lowers a loop invariant to a closure call, and its loop bodies write
   enough temporaries that the k-induction havoc block in front of the loop
   head is longer than the window the invariant marker was searched in. */
unsigned nondet_uint(void);
struct env
{
  unsigned *i, *n;
  unsigned long long *s;
};
_Bool closure(struct env *c)
{
  return *c->i <= *c->n && *c->s == 2ull * *c->i;
}
int main()
{
  unsigned n = nondet_uint(), i = 0;
  unsigned long long s = 0;
  unsigned t0, t1, t2, t3, t4, t5, t6, t7;
  struct env c = {&i, &n, &s};
  while (i < n)
    __CPROVER_loop_invariant(closure(&c))
  {
    t0 = i;
    t1 = t0 + 1;
    t2 = t1;
    t3 = t2;
    t4 = t3;
    t5 = t4;
    t6 = t5;
    t7 = t6;
    i = t7;
    s += 2;
  }
  __CPROVER_assert(s == 2ull * n, "s is twice n");
  return 0;
}
