extern int __VERIFIER_nondet_int(void);

// Companion to github_6031: a terminating program under the same flag
// combination (--termination --k-induction-parallel). The loop always
// terminates, so the sequential termination strategy (which --termination
// now routes to instead of the parallel driver) must report VERIFICATION
// SUCCESSFUL. This pins that the priority routing is not a blanket "always
// FAILED".
int main()
{
  int n = __VERIFIER_nondet_int();
  while (n < 10)
    n++;
  return 0;
}
