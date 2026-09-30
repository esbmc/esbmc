// A write through a pointer resets the interval domain to the shared empty map.
// y = 3 must not be written into that shared map, or the second reset would
// bring the stale interval back and prune the claim.
int nondet_int();
int main()
{
  int y, z;
  int *p = &y;
  z = 1;
  y = 3;
  *p = nondet_int();
  y = 3;
  *p = nondet_int();
  __ESBMC_assert(y == 3, "y still 3");
}
