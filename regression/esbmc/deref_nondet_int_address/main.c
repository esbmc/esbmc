unsigned long nondet_ulong();

int main()
{
  int *p = (int *)nondet_ulong();
  return *p;
}
