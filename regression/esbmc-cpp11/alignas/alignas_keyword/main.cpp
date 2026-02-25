#include <assert.h>
#include <stdint.h>
#include <stddef.h>

// alignas(max_align_t) types (emulate default align behaviour)

// Struct with alignment expression
struct alignas(max_align_t) default_alignas_aligned_struct
{
  char data;
};

// Union with alignment expression
union alignas(max_align_t) default_alignas_aligned_union
{
  int value;
  char bytes[4];
};

// Variable with alignment
alignas(max_align_t) int default_global_var = 42;

// alignas(expr) types

// Struct with alignment expression
struct alignas(16) expr_alignas_aligned_struct
{
  char data;
};

// Union with alignment expression
union alignas(32) expr_alignas_aligned_union
{
  int value;
  char bytes[4];
};

// Variable with alignment
alignas(64) int expr_global_var = 42;

// alignas(type) types

// Struct with alignment expression
struct alignas(long double) type_alignas_aligned_struct
{
  char data;
};

// Union with alignment expression
union alignas(long long) type_alignas_aligned_union
{
  int value;
  char bytes[4];
};

// Variable with alignment
alignas(long long) int type_global_var = 42;

// Test: alignas(max_align_t) (emulate default align behaviour)
void test_default()
{
  // Local variable with alignment
  alignas(max_align_t) int local_var = 123;

  struct default_alignas_aligned_struct s;
  union default_alignas_aligned_union u;

  s.data = 'A';
  u.value = 100;

  assert(s.data == 'A');
  assert(u.value == 100);
  assert(default_global_var == 42);
  assert(local_var == 123);
  assert((uintptr_t)&s % alignof(decltype(s)) == 0);
  assert((uintptr_t)&u % alignof(decltype(u)) == 0);
  assert((uintptr_t)&default_global_var % alignof(decltype(default_global_var)) == 0);
  assert((uintptr_t)&local_var % alignof(decltype(local_var)) == 0);
}

// Test: alignas(expr)
void test_expr()
{
  // Local variable with alignment
  alignas(16) int local_var = 123;

  struct expr_alignas_aligned_struct s;
  union expr_alignas_aligned_union u;

  s.data = 'A';
  u.value = 100;

  assert(s.data == 'A');
  assert(u.value == 100);
  assert(expr_global_var == 42);
  assert(local_var == 123);
  assert((uintptr_t)&s % alignof(decltype(s)) == 0);
  assert((uintptr_t)&u % alignof(decltype(u)) == 0);
  assert((uintptr_t)&expr_global_var % alignof(decltype(expr_global_var)) == 0);
  assert((uintptr_t)&local_var % alignof(decltype(local_var)) == 0);
}

// Test: alignas(type)
void test_type()
{
  // Local variable with alignment
  alignas(type_alignas_aligned_struct) int local_var = 123;

  struct type_alignas_aligned_struct s;
  union type_alignas_aligned_union u;

  s.data = 'A';
  u.value = 100;

  assert(s.data == 'A');
  assert(u.value == 100);
  assert(type_global_var == 42);
  assert(local_var == 123);
  assert((uintptr_t)&s % alignof(decltype(s)) == 0);
  assert((uintptr_t)&u % alignof(decltype(u)) == 0);
  assert((uintptr_t)&type_global_var % alignof(decltype(type_global_var)) == 0);
  assert((uintptr_t)&local_var % alignof(decltype(local_var)) == 0);
}

int main()
{
  test_default();
  test_expr();
  test_type();
  return 0;
}
