/* A global initialized by a side-effecting call lowers, inside the synthesized
 * __ESBMC_main body, to an ASSIGN whose location comes from
 * restore_value_locations. That block is unlocated, so the walk used to abandon
 * the whole subtree and the instruction came out with no location at all. */
int nondet_int();

int A = nondet_int();

int main()
{
  return 0;
}
