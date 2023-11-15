/* reduced version from SV-COMP 2023: c/loop-invgen/string_concat-noarr.i */

extern void abort(void);

extern void __assert_fail (const char *__assertion, const char *__file,
      unsigned int __line, const char *__function)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__noreturn__));
extern void __assert_perror_fail (int __errnum, const char *__file,
      unsigned int __line, const char *__function)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__noreturn__));
extern void __assert (const char *__assertion, const char *__file, int __line)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__noreturn__));

void reach_error() { ((void) sizeof ((0) ? 1 : 0), __extension__ ({ if (0) ; else __assert_fail ("0", "assert.h", 3, __extension__ __PRETTY_FUNCTION__); })); }
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
int __VERIFIER_nondet_int();
int main(void) {
  int i, j;
 L0:
  i = 0;
 L1:
  while( __VERIFIER_nondet_int() && i < 3){
    i++;
  }
  if(i >= 1) STUCK: goto STUCK;
  j = 0;
 L2:
  while( __VERIFIER_nondet_int() && i < 3 ){
    i++;
    j++;
  }
  if(j >= 1) goto STUCK;
  __VERIFIER_assert( i < 2 );
  return 0;
}
