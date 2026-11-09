/* #9937: a byte write next to an out-of-bounds pointer in a union. */
int main()
{
  union { int *p; struct { int *a[1]; char c[2]; } s; } u;
  int arr[2];
  u.p = arr + 100;
  u.s.c[1] = 4;
  __ESBMC_assert(0, "show");
}
