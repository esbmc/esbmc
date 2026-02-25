#include <assert.h>
#include <stdint.h>

// __attribute__((aligned)) (default value) types

// Struct with alignment expression
struct __attribute__((aligned)) default_aligned_struct
{
  char data;
};

// Union with alignment expression
union __attribute__((aligned)) default_aligned_union
{
  int value;
  char bytes[4];
};

// Variable with alignment
__attribute__((aligned)) int default_global_var = 42;

// __attribute__((aligned(expr))) types

// Struct with alignment expression
struct __attribute__((aligned(16))) expr_aligned_struct
{
  char data;
};

// Union with alignment expression
union __attribute__((aligned(32))) expr_aligned_union
{
  int value;
  char bytes[4];
};

// Variable with alignment
__attribute__((aligned(64))) int expr_global_var = 42;

// Test: __attribute__((aligned)) (default value)
void test_default_fail()
{
  // Local variable with alignment
  __attribute__((aligned)) int local_var = 123;

  struct default_aligned_struct s;
  union default_aligned_union u;

  s.data = 'A';
  u.value = 100;

  assert(s.data == 'A');
  assert(u.value == 100);
  assert(default_global_var == 42);
  assert(local_var == 123);
  assert((uintptr_t) &s % _Alignof(typeof(s)) == 0);
  assert((uintptr_t) &u % _Alignof(typeof(u)) == 0);
  assert((uintptr_t) &default_global_var % _Alignof(typeof(default_global_var)) == 0);
  assert((uintptr_t) &local_var % _Alignof(typeof(local_var)) != 0); // Should be zero
}

// Test: __attribute__((aligned(expr)))
void test_expr_fail()
{
  // Local variable with alignment
  __attribute__((aligned(16))) int local_var = 123;

  struct expr_aligned_struct s;
  union expr_aligned_union u;

  s.data = 'A';
  u.value = 100;

  assert(s.data == 'A');
  assert(u.value == 100);
  assert(expr_global_var == 42);
  assert(local_var == 123);
  assert((uintptr_t) &s % _Alignof(typeof(s)) != 0);
  assert((uintptr_t) &u % _Alignof(typeof(u)) == 0);
  assert((uintptr_t) &local_var % _Alignof(typeof(local_var)) == 0);
  assert((uintptr_t) &expr_global_var % _Alignof(typeof(expr_global_var)) != 0); // Should be zero
}

int main()
{
  test_default_fail();
  test_expr_fail();
  return 0;
}
