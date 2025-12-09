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
int __VERIFIER_nondet_int();
int main() {
   int n = 0;
   int j = 1;
   int k = __VERIFIER_nondet_int();
  __ESBMC_assume(k == j * -1);
  __VERIFIER_assert(k == (2 - 1) * j * -1);
}
