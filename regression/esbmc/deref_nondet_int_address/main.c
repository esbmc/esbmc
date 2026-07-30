unsigned long nondet_ulong(void);

int main()
{
  // The same shape with a nondet address is caught; pins the boundary of #6544
  // rather than the integer-to-pointer cast alone.
  int *p = (int *)nondet_ulong();
  return *p;
}
