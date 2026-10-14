unsigned long nondet_ulong(void);

int main()
{
  // The nondet counterpart of deref_constant_int_address (#6544).
  int *p = (int *)nondet_ulong();
  return *p;
}
