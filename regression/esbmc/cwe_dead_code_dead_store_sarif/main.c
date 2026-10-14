// --dead-code-check and --dead-store-check share the single --sarif-output
// path. Each advisory used to write its own SARIF document, so the second write
// truncated the first away (or, with "-", emitted two concatenated JSON
// documents). Both sets must land in one run (issue #4495).
int main(void)
{
  int x = 1;
  int unused = 42; // dead store, never read (CWE-563)
  if (x)
    return 0;
  else
    return 1; // dead branch, x is always 1 (CWE-561)
}
