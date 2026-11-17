extern double __VERIFIER_nondet_double(void);
extern int __VERIFIER_nondet_int(void);
int main()
{
  double a = __VERIFIER_nondet_double();
  double b = a * 10.0;
  int i = __VERIFIER_nondet_int();
  int j = i + 1;
  return (int)(b != b) + j;
}
