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

  /* Subtraction, where the widening bound is least obvious: unsigned max
     minus signed min needs the exact type to hold both ends. Proved against
     the same wide reference as the addition above, over all int pairs. */
  long long diff = (long long)a - (long long)b;
  int sr;
  assert(__builtin_sub_overflow(a, b, &sr) ==
         (diff < INT_MIN || diff > INT_MAX));
  assert(sr == (int)diff);
  assert(!__builtin_sub_overflow(a, a, &sr) && sr == 0);
  assert(__builtin_sub_overflow(INT_MIN, 1, &sr));
  assert(!__builtin_sub_overflow(0u, 1u, &sr) && sr == -1);

  unsigned char uc;
  assert(__builtin_sub_overflow(0, 1, &uc));

  /* clang accepts a _Bool result and stores the exact value truncated to one
     bit, so 1 + 1 stores 0 rather than the 1 a cast to _Bool would give. */
  _Bool bres;
  assert(__builtin_add_overflow(1, 1, &bres) && bres == 0);
  assert(!__builtin_add_overflow(0, 1, &bres) && bres == 1);

  return 0;
}
