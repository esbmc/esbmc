extern void abort(void);
extern void __assert_fail(const char *, const char *, unsigned int, const char *) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__noreturn__));
void reach_error() { __assert_fail("0", "vnew2.c", 3, "reach_error"); }
extern void abort(void);
void assume_abort_if_not(int cond) {
  if(!cond) {abort();}
}
void __VERIFIER_assert(int cond) {
  if (!(cond)) {
    ERROR: {reach_error();abort();}
  }
  return;
}

int SIZE = 20000001;
unsigned int __VERIFIER_nondet_uint();
int main() {
  unsigned int n,i,k,j;
  n = __VERIFIER_nondet_uint();
  i = __VERIFIER_nondet_uint();
//  if (!(n <= SIZE)) return 0;
  j = i;
  k = j;

  // __invariant(i == j && j == k);
  while( i < n ) {
    i = i + 3;
    j = j+3;
    k = k+3;
    if(i == 3) break;
  }
  if(n>0)
	  __VERIFIER_assert( i==j && j==k );
}

