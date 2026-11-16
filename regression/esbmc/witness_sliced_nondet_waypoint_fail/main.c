// Only the last call's result decides the violation, so the slicer drops the
// first two. A validator matches a function_return waypoint to the next call
// at line 11, so the witness must name all three calls.
extern unsigned __VERIFIER_nondet_uint(void);
extern void reach_error(void);

int main(void)
{
  unsigned v = 0;
  for (int i = 0; i < 3; i++)
    v = __VERIFIER_nondet_uint();
  if (v == 7)
    reach_error();
  return 0;
}
