double nondet_double();

int main()
{
  double x = nondet_double();
  assert(x==x);
  return 0;
}
