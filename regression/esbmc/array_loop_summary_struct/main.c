/* A struct element is left to unwinding: a lambda over a struct array
   reaches the solver's tuple flattening, which aborted. */
typedef struct { double c[2]; int k; } P;
P pts[3];
int nondet_int(void);
int main()
{
  P old[3];
  for (int i = 0; i < 3; i++)
    old[i] = pts[i];
  int x = nondet_int();
  __ESBMC_assume(x >= 0 && x < 3);
  old[x].k++;
  int y = nondet_int();
  __ESBMC_assume(y >= 0 && y < 3);
  __ESBMC_assert(old[y].k <= 1, "");
  return 0;
}
