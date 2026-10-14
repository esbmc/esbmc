// --dead-code-check probes conditional branch *directions* only. A statement
// after an unconditional return is a textbook CWE-561 case but is out of scope
// (no branch guard to probe), so the analysis reports no dead code here.
// Documents the intentional scope limitation.
int main(void)
{
  return 0;
  return 1;
}
