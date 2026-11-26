/* goto-cc gives the do-while and the inner loop one head (#11085). */
struct env { unsigned *i; };
_Bool inv(struct env *c, int k) { return *c->i <= 4 && k == 1; }
int main(void)
{
  unsigned i = 0, j = 0;
  struct env c = {&i};
  do
    __CPROVER_loop_invariant(inv(&c, 1) != 0 && 1)
    {
      while (j < 3)
        j++;
      if (i >= 5)
        break;
      i += 1;
    }
  while (1);
  __CPROVER_assert(i == 5, "i is 5");
  return 0;
}
