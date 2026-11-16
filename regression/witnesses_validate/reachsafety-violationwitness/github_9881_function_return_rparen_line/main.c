/* #9881: a function_return waypoint names a call by the line of its ')'. */
extern int __VERIFIER_nondet_int(void);
int value;
int main(void)
{
  // clang-format off
  value = __VERIFIER_nondet_int(
  );
  // clang-format on
  __ESBMC_assert(value != 1, "target");
}
