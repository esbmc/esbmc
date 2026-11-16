/* Issue #9689: the YAML witness must not carry one branching waypoint per
   executed branch. */
unsigned nondet_uint(void);
int main(void)
{
  unsigned n = nondet_uint();
  int s = 0;
  for (int i = 0; i < 1500; i++)
    if ((i ^ n) % 2)
      s++;
  __ESBMC_assert(s != 750, "reached");
  return 0;
}
