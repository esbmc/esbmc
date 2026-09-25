/* replace_call_branch_target_pass (issue #6364):
 * A contracted call sitting at a branch-target position (the first instruction
 * of the `else` branch) must keep its contract replacement. Before the fix the
 * replacement was spliced *before* the call and the incoming jump landed on the
 * post-replacement SKIP, so on the else path fb's contract (which sets b_done)
 * was dropped and b_done was left unconstrained, spuriously failing the assert.
 *
 * Expected: VERIFICATION SUCCESSFUL
 */
typedef struct
{
  int received;
} T;
_Bool a_done = 0, b_done = 0;

void fa(T *self, int v)
{
  __ESBMC_requires(self != 0);
  __ESBMC_assigns(self->received, a_done);
  __ESBMC_ensures(self->received == v);
  __ESBMC_ensures(a_done == 1);
  self->received = v;
  a_done = 1;
}

void fb(T *self, int v)
{
  __ESBMC_requires(self != 0);
  __ESBMC_assigns(self->received, b_done);
  __ESBMC_ensures(self->received == v);
  __ESBMC_ensures(b_done == 1);
  self->received = v;
  b_done = 1;
}

int main(void)
{
  T t1, t2;
  int c;
  __ESBMC_assume(c == 0 || c == 1);
  if (c == 0)
  {
    fb(&t2, 2);
    fa(&t1, 1);
  }
  else
  {
    fb(&t2, 2);
    fa(&t1, 1);
  }
  if (a_done)
    __ESBMC_assert(b_done, "b_done must hold whenever a_done does");
  return 0;
}
