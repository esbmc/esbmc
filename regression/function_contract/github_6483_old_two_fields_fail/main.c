/* github_6483_old_two_fields_fail:
 *   Two __ESBMC_old snapshots of distinct fields of the same is_fresh
 *   heap-backed struct, with the body writing both. The second ensures clause
 *   is violated (count gains 2, the contract promises 1) and must be caught.
 *
 *   This was a false negative: the byte-level CONCAT/byte_extract encoding of
 *   the two overlapping field updates lost the violation and the wrapper
 *   reported VERIFICATION SUCCESSFUL (#6483). Fixed before this test landed;
 *   the test pins it, since nothing else covers the shape.
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
  c->count += 2;
}

int main()
{
  return 0;
}
