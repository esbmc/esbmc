#include <assert.h>
#include <limits.h>

int nondet_int(void);
unsigned nondet_uint(void);

/* The type-generic __builtin_{add,sub,mul}_overflow, unlike the typed
 * __builtin_sadd_overflow family, reach ESBMC as calls clang has not lowered.
 * Left unmodelled they returned nondet with a warning, so UInt<128> 1 + 1
 * came back zero-with-overflow in llvm-libc's big_int.h.
 *
 * GCC performs the operation as if in infinite precision, so the operands
 * must not be narrowed to the result type before the check: (signed char)200
 * + 200 overflows a signed char even though the ints add fine. */
int main(void)
{
  int a = nondet_int(), b = nondet_int();
  long long exact = (long long)a + (long long)b;
  int r;
  assert(__builtin_add_overflow(a, b, &r) ==
         (exact < INT_MIN || exact > INT_MAX));
  assert(r == (int)exact);

  unsigned c = nondet_uint(), d = nondet_uint();
  unsigned long long prod = (unsigned long long)c * d;
  unsigned ur;
  assert(__builtin_mul_overflow(c, d, &ur) == (prod > UINT_MAX));
  assert(ur == (unsigned)prod);

  /* Operands and result of differing type and signedness. Each call needs its
   * own instantiation: keying the mangled name on the first parameter alone
   * gave every call the first one's result type. */
  signed char sc;
  assert(__builtin_add_overflow(200, 200, &sc));

  long long ll;
  assert(!__builtin_mul_overflow(100000, 100000, &ll) && ll == 10000000000LL);

  unsigned u;
  assert(!__builtin_add_overflow(-1, 1, &u) && u == 0u);
  assert(__builtin_add_overflow(-1, 0, &u));

  return 0;
}
