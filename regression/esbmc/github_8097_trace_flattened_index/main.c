/* #8097: an element of a two-dimensional array read back from a union's bits
 * for the counterexample is selected row by row, not by its flattened index. */
int nondet_int();
int main()
{
  union
  {
    unsigned long long l[6];
    int m[3][4];
  } u;
  u.m[0][2] = nondet_int();
  u.m[1][2] = nondet_int();
  __ESBMC_assert(u.m[0][2] != 3 || u.m[1][2] != 5, "show");
}
