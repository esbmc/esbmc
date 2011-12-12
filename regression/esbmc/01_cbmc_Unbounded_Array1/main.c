int main() {
  unsigned int n, i, j, ai, aj;
  int a[n];
  
  __ESBMC_assume(n>10 && n<10000000);
   
  __ESBMC_assume(i<n);
  __ESBMC_assume(j<n);
   
  ai=a[i];
  aj=a[j];

  assert(ai==aj || i!=j);
}
