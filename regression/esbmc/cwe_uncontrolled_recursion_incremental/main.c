// Documents the interaction with the incremental strategy: under
// --incremental-bmc the recursion-unwinding counterexample is not produced the
// same way as plain --unwind BMC, so no CWE-674 line is emitted. This pins the
// "inherent, not a regression" contract from the PR: a future change to the
// inductive-step / incremental handling must not silently start (or stop)
// emitting CWE-674 here without updating this test.
int f(int n)
{
  return f(n + 1);
}

int main(void)
{
  return f(0);
}
