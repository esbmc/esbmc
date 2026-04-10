extern int __VERIFIER_nondet_int(void);

int main() 
{
  int n = __VERIFIER_nondet_int();
  while (n < 10 ) 
  {
    __VERIFIER_assume(n >= 0);
    n++;
  }

  return 0;
}
