// #9937: a pointer read back after a byte write over part of its bits, then
// stored again, must not make the path vacuous.
int main()
{
  union { char c; int *p; } u;
  int arr[2];
  u.p = arr + 100;
  u.c = 1;
  int *q = u.p;
  u.p = q;
  __ESBMC_assert(0, "reachable");
}
