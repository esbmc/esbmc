extern void abort(void);
extern void __assert_fail(const char *, const char *, unsigned int, const char *) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__noreturn__));
void reach_error() { __assert_fail("0", "inv_Newton-2.c", 3, "reach_error"); }
/* Example from "Static Analysis of Numerical Algorithms", by
   Goubault and Putot. Published in SAS 06.

   Computing 1/x using Newton's method.
*/

/*
 * This task results in "false" due to the types of
 * the members of the union "double_int".
 *
 * The algorithm in function inv(double) tries to
 * extract the parameter's exponent but instead gets
 * some arbitrary bits of its significant.
 */
extern double __VERIFIER_nondet_double();
extern void abort(void);
void assume_abort_if_not(int cond) {
  if(!cond) {abort();}
}
void __VERIFIER_assert(int cond) { if (!(cond)) { ERROR: {reach_error();abort();} } return; }

union double_int
{
  double d;
  int i;
};

double inv (double A)
{
  double xi, xsi, temp;
  signed int cond, exp;
  union double_int A_u, xi_u;
  A_u.d = A;
  exp = (signed int) ((A_u.i & 0x7FF00000) >> 20) - 1023;
  xi_u.d = 1;
  xi_u.i = ((1023-exp) << 20);
  xi = xi_u.d;
  cond = 1; 
  while (cond) {
    xsi = 2*xi-A*xi*xi; 
    temp = xsi-xi;
    cond = ((temp > 1e-10) || (temp < -1e-10));
    xi = xsi; 
  }
  return xi;
}

int main()
{
  double a,r;

  a = __VERIFIER_nondet_double();
  assume_abort_if_not(a >= 20. && a <= 30.);

  r = inv(a);

  __VERIFIER_assert(r >= 0 && r <= 0.06);
  return 0;
}
