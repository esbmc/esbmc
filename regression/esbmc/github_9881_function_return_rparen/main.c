/* #9881: a function_return waypoint sits at the call's ')' and names the
 * value the call returned (3), not the value of x (7). A call assigned
 * directly (y) gets one too. */
extern int __VERIFIER_nondet_int(void);
extern unsigned char __VERIFIER_nondet_uchar(void);

int main(void)
{
  int x = 1 + 2 * __VERIFIER_nondet_uchar();
  int y;
  y = __VERIFIER_nondet_int();
  __ESBMC_assert(x != 7 || y != 5, "x is 7 and y is 5");
}
