// The one input in the 3355-test C/C++ corpus that still takes the whole-body
// fallback (W1-loc, esbmc/esbmc#4715). remove_sideeffects on a for-iteration
// holding a call with a side-effecting argument allocates a temp, whose
// convert_decl pushes a code_dead; the native for-arm's destructor-stack
// invariance check then declines. Pins that the fallback stays byte-identical
// for it -- reduced from cbmc/01_cbmc_for4.
extern void acall(int a);

int main(void)
{
  int i;
  for (i = 0; i < 3; acall(i++))
  {
  }
  return i;
}
