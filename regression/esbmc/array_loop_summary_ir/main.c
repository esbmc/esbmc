// --ir does not wrap the closed form's unsigned arithmetic: x started at -1.
int a[8]; int nondet_int();
int main(){ int n=nondet_int(); __ESBMC_assume(n>0&&n<=8);
  int x=-1;
  for(int i=0;i<n;i++){ a[i]=0; x=x+1; }
  __ESBMC_assert(x != n-1, ""); }
