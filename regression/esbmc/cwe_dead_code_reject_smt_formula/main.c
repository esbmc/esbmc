// --dead-code-check with --smt-formula-only emits the SMT formula (P_SMTLIB)
// without solving any probe, so every branch would be misreported as dead
// (issue #5934). The combination must be rejected up front.
int main(void)
{
  return 0;
}
