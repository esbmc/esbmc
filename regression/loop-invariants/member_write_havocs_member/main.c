/* s->flag is the only storage the loop writes: the rest of *s keeps its value. */
struct S { int n; char flag; int other; };
void f(struct S *s)
{
  __ESBMC_requires(__ESBMC_is_fresh(s, sizeof(*s)) && s->n >= 0 && s->n <= 1000000);
  __ESBMC_assigns(s->flag);
  __ESBMC_ensures(s->other == __ESBMC_old(s->other) && s->flag == (s->n > 0));
  s->flag = 0;
  int i = 0;
  __ESBMC_loop_invariant(0 <= i && i <= s->n && s->flag == (i > 0));
  for (; i < s->n; i++)
    s->flag = 1;
}
