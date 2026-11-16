/* #9881: after the if, g's phi merge takes the location of the next
 * instruction, the call assigned to u. Only the call's value (7) gets a
 * function_return waypoint there, not g's value (3). */
extern int __VERIFIER_nondet_int(void);
int g;
void set_g(int v)
{
  g = v + 1;
}
int main(void)
{
  int u;
  int t = __VERIFIER_nondet_int();
  if (t)
    set_g(t);
  u = __VERIFIER_nondet_int();
  __ESBMC_assert(!(t && g == 3 && u == 7), "u is 7");
}
