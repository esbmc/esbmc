/* github_6483_old_two_fields_pass:
 *   The positive control for github_6483_old_two_fields_fail: same shape, but
 *   the body honours both clauses. Guards against "fixing" the false negative
 *   by making every such contract fail (#6483).
 */
typedef struct
{
  int val;
  int count;
} Counter;

void f(Counter *c)
{
  __ESBMC_requires(__ESBMC_is_fresh(c, sizeof(Counter)));
  __ESBMC_ensures(c->val == __ESBMC_old(c->val) + 1);
  __ESBMC_ensures(c->count == __ESBMC_old(c->count) + 1);
  c->val++;
  c->count++;
}

int main()
{
  return 0;
}
