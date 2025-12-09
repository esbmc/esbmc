float nondet_float();

int main()
{
  float x = nondet_float();
  assert(x==x);
  return 0;
}
