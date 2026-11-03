// Padded to powers of two, this array has 2^64 elements: rows must not alias
#define K 16385
unsigned nondet_uint(void);
char a[16][K][K][K][K];
int main(void){
  unsigned i = nondet_uint(); __ESBMC_assume(i < 16);
  a[0][0][0][0][0] = 1;
  __ESBMC_assert(a[i][0][0][0][0] == (i == 0), "distinct rows");
  return 0;
}
