// Guards the fold against over-reach: int -> _Bool -> int is not the identity,
// so this must stay refutable. If a future simplifier folds the reverse trip
// away, this flips to SUCCESSFUL. #4626.
int nondet_int(void);

int main(void)
{
  int i = nondet_int();
  __ESBMC_assert((int)(_Bool)i == i, "reverse round trip is not the identity");
  return 0;
}
