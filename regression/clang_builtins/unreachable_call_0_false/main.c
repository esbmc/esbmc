// This function should always receive (0, 1 or 2)
int foo(int a)
{
  switch(a)
  {
  case 0:
    return 0;
  case 1:
    return 1;
  case 2:
    return 2;
  default:
    __builtin_unreachable();
  }
}


int main() {
  unsigned a = __VERIFIER_nondet_uint() % 5;
  foo(a);
  return 0;
}
