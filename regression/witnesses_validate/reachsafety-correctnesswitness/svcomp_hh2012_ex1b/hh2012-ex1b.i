
extern void __assert_fail (const char *__assertion, const char *__file,
      unsigned int __line, const char *__function)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__noreturn__));
extern void __assert_perror_fail (int __errnum, const char *__file,
      unsigned int __line, const char *__function)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__noreturn__));
extern void __assert (const char *__assertion, const char *__file, int __line)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__noreturn__));

extern void abort(void);
void reach_error() { ((void) sizeof ((0) ? 1 : 0), __extension__ ({ if (0) ; else __assert_fail ("0", "hh-ex1b.c", 3, __extension__ __PRETTY_FUNCTION__); })); }
void __VERIFIER_assert(int cond) { if(!(cond)) { ERROR: {reach_error();abort();} } }
int main() {
  int i = 0;
  while (i < 100) {
    int j = 0;
    while (j < 100) {
      j++;
      __VERIFIER_assert(j <= 100);
    }
    __VERIFIER_assert(j == 100);
    i++;
    __VERIFIER_assert(i <= 100);
  }
  __VERIFIER_assert(i == 100);
  return 0;
}
