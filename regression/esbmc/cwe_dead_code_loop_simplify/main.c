// The empty constant-false loop below is erased by goto_loop_simplify in a
// normal run. --dead-code-check must keep it (like the other coverage modes)
// so its provably-dead loop-entry direction is still probed and reported
// CWE-561. Pins the goto_loop_simplify gate: without it this branch vanishes
// before instrumentation and the dead code is silently missed.
int main(void)
{
  for (int i = 0; i < 0; i++)
  {
  }
  return 0;
}
