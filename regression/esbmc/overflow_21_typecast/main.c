int nondet_double();

int main()
{
  int a;
  double b=nondet_double();

  a = (int)b;
}
