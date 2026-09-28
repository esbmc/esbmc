/* The cursor is reached through a pointer parameter rather than being global,
 * so `in_pre_state` resolves nothing for it -- `collect_global_variables`
 * skips pointer types. Moving it first writes buf[*p + 1], which the clause
 * never granted, and reading the index back after the body excused it. */
void push_d(int *buf, int *p, int v)
{
  __ESBMC_requires(__ESBMC_is_fresh(buf, 8 * sizeof(int)));
  __ESBMC_requires(__ESBMC_is_fresh(p, sizeof(int)));
  __ESBMC_requires(*p >= 0 && *p < 7);
  __ESBMC_assigns(buf[*p], *p);
  __ESBMC_ensures(1);
  *p = *p + 1;
  buf[*p] = v;
}

int main()
{
  return 0;
}
