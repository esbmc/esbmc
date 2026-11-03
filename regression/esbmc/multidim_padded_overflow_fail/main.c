// Padded to powers of two, this array has 2^64 elements: rows must not alias
#define K 16385
unsigned nondet_uint(void);
char a[16][K][K][K][K];
int main(void){
  unsigned i = nondet_uint(), z = nondet_uint();
  __ESBMC_assume(i == 1 && z == 0);
  a[z][0][0][0][0] = 1;
  a[i][0][0][0][0] = 2;
  __ESBMC_assert(a[z][0][0][0][0] == 2, "a[0] holds 1");
  return 0;
}
