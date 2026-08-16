/* github_6484_unmentioned_ptr_param_pass:
 * q is a pointer parameter the assigns clause never mentions and the body
 * never touches. Phase 2C snapshots *p for every such parameter to prove it
 * unchanged, but q's harness backing has a nondet extent, so that
 * harness-invented read is itself out of bounds. The wrapper would then report
 * a violation in a parameter the contract says nothing about.
 *
 * Params with an unjustified backing must be skipped: the body cannot validly
 * dereference them either, so there is nothing for the frame check to protect.
 */
void f(int *p, int *q)
{
  __ESBMC_requires(p != 0);
  __ESBMC_requires(q != 0);
  __ESBMC_requires(__ESBMC_is_fresh(p, sizeof(int)));
  __ESBMC_assigns(*p);
  __ESBMC_ensures(*p == 1);
  *p = 1;
}

int main()
{
  return 0;
}
