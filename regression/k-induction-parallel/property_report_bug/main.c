/* The per-property report, and the CWE line its rows carry, must appear under
   --k-induction-parallel too: the base-case child holds the verdicts and is
   the process whose result decides the run. */
int main(void)
{
  int *p = 0;
  *p = 42;
  return 0;
}
