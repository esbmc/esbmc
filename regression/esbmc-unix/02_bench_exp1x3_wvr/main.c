typedef unsigned long int pthread_t;

union pthread_attr_t
{
  char __size[36];
  long int __align;
};
typedef union pthread_attr_t pthread_attr_t;

extern void __assert_fail(const char *__assertion, const char *__file,
      unsigned int __line, const char *__function)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__noreturn__));
void reach_error() { __assert_fail("0", "bench-exp1x3.wvr.c", 21, __extension__ __PRETTY_FUNCTION__); }
extern int pthread_create (pthread_t *__restrict __newthread,
      const pthread_attr_t *__restrict __attr,
      void *(*__start_routine) (void *),
      void *__restrict __arg) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 3)));

extern unsigned int  __VERIFIER_nondet_uint(void);


unsigned int x1, x2, n;

void* thread1() {
  while (x1 == 1){
    x2 = 1;
  }
  return 0;
}

int main() {
  pthread_t t1, t2;
  
  x1 = __VERIFIER_nondet_uint();
  x2 = __VERIFIER_nondet_uint();
  
  if(x1 != x2){
   return 0;;     
  }

  pthread_create(&t1, 0, thread1, 0);
  if(x1 != x2){
   __assert_fail("0", "bench-exp1x3.wvr.c", 44, __extension__ __PRETTY_FUNCTION__);
  }
  return 0;
}
