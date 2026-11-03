/* Issue #9689: the YAML witness must not carry one branching waypoint per
   executed branch. */
int main(void)
{
  int s = 0;
  for (int i = 0; i < 1500; i++)
    if (i % 2)
      s++;
  __ESBMC_assert(s == 750, "reached");
  return 0;
}
