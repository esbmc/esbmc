
extern void abort(void);
extern void __assert_fail(const char *, const char *, unsigned int, const char *) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__noreturn__));
void reach_error() { __assert_fail("0", "linear_search.c", 3, "reach_error"); }
extern void *calloc(unsigned int num, unsigned int size);
void __VERIFIER_assert(int cond) {
  if (!(cond)) {
    ERROR: {reach_error();abort();}
  }
  return;
}
unsigned int __VERIFIER_nondet_uint();
unsigned int  SIZE;
const unsigned int MAX = 100000;
int linear_search(int *a, int n, int q) {
  unsigned int j=0;
  while (j<n && a[j]!=q) {
  j++;
  if (j==20) j=-1;
  }
  if (j<SIZE) return 1;
  else return 0;
}
int main() { 
  SIZE=(__VERIFIER_nondet_uint()/2)+1;
  if (SIZE > 1 && SIZE < MAX) {
    int *a = calloc(SIZE,sizeof(int));
    a[SIZE/2]=3;
    __VERIFIER_assert(linear_search(a,SIZE,3));
  }
}
