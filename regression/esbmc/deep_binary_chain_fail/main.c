/* Same depth as deep_binary_chain_pass, but reachable and not valid for every
 * x. A crash-only test would still pass if the conversion survived yet built
 * the wrong expression; this one would not. */
#define A1 x && x && x && x
#define A2 A1 && A1 && A1 && A1
#define A3 A2 && A2 && A2 && A2
#define A4 A3 && A3 && A3 && A3
#define A5 A4 && A4 && A4 && A4
#define A6 A5 && A5 && A5 && A5
#define DEEP A6 && A6 && A6

int main(void)
{
  int x = nondet_int();
  __ESBMC_assert(DEEP, "a deep conjunction does not hold for x == 0");
  return 0;
}
