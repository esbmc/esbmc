/* `b` is unconstrained, so the conversion below is undefined whenever its
   integral part does not fit an int, NaN included (C11 6.3.1.4p1). This
   expected SUCCESSFUL until #7572 added the check that catches it. */
int main()
{
  int a;
  double b=nondet_double();

  a = (int)b;
}
