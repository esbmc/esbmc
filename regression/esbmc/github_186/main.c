int main() 
{
  int a = __VERIFIER_nondet_int();
  __VERIFIER_assume(a >= 0 && a <= 4);

  int x1[5][5] = {0};
  x1[2][a] = 2;
  assert(x1[2][a] == 2);
} 
