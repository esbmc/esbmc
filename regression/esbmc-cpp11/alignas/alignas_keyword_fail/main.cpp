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

// Struct with aligned field
struct default_field_aligned
{
  char prefix;
  alignas(max_align_t) int aligned_field;
  char suffix;
};

// Variable with alignment
alignas(max_align_t) int default_global_var = 42;

// alignas(expr) types

// Struct with alignment expression
struct alignas(16) expr_alignas_aligned_struct
{
  char data;
};

// Struct with aligned field
struct expr_field_aligned
{
  char prefix;
  alignas(8) int aligned_field;
  char suffix;
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

// Struct with aligned field
struct type_field_aligned
{
  char prefix;
  alignas(float) int aligned_field;
  char suffix;
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
void test_default_fail()
{
  // Local variable with alignment
  alignas(max_align_t) int local_var = 123;

  struct default_alignas_aligned_struct s;
  union default_alignas_aligned_union u;
  struct default_field_aligned f;

  s.data = 'A';
  u.value = 100;
  f.aligned_field = 200;

  assert(s.data == 'A');
  assert(u.value == 100);
  assert(f.aligned_field == 200);
  assert(default_global_var == 42);
  assert(local_var == 123);
  assert((uintptr_t)&s % alignof(decltype(s)) == 0);
  assert((uintptr_t)&u % alignof(decltype(u)) == 0);
  assert((uintptr_t)&f.aligned_field % alignof(decltype(f)) == 0); // The struct should be aligned, but this isn't guaranteed right now
  // see "alignas_keyword_default_aligned_field_fail" test for the case where we check that the struct is also aligned
  assert((uintptr_t)&default_global_var % alignof(decltype(default_global_var)) == 0);
  assert(
    (uintptr_t)&local_var % alignof(decltype(local_var)) != 0); // Should be zero
}

// Test: alignas(expr)
void test_expr_fail()
{
  // Local variable with alignment
  alignas(16) int local_var = 123;

  struct expr_alignas_aligned_struct s;
  union expr_alignas_aligned_union u;
  struct expr_field_aligned f;

  s.data = 'A';
  u.value = 100;
  f.aligned_field = 200;

  assert(s.data == 'A');
  assert(u.value == 100);
  assert(f.aligned_field == 200);
  assert(expr_global_var == 42);
  assert(local_var == 123);
  assert((uintptr_t)&s % alignof(decltype(s)) == 0);
  assert((uintptr_t)&u % alignof(decltype(u)) == 0);
  assert((uintptr_t)&f.aligned_field % alignof(decltype(f)) == 0); // The struct should be aligned, but this isn't guaranteed right now
  // see "alignas_keyword_expr_aligned_field_fail" test for the case where we check that the struct is also aligned
  assert((uintptr_t)&expr_global_var % alignof(decltype(expr_global_var)) == 0);
  assert(
    (uintptr_t)&local_var % alignof(decltype(local_var)) != 0); // Should be zero
}

// Test: alignas(type)
void test_type_fail()
{
  // Local variable with alignment
  alignas(type_alignas_aligned_struct) int local_var = 123;

  struct type_alignas_aligned_struct s;
  union type_alignas_aligned_union u;
  struct type_field_aligned f;

  s.data = 'A';
  u.value = 100;
  f.aligned_field = 200;

  assert(s.data == 'A');
  assert(u.value == 100);
  assert(f.aligned_field == 200);
  assert(type_global_var == 42);
  assert(local_var == 123);
  assert((uintptr_t)&s % alignof(decltype(s)) == 0);
  assert((uintptr_t)&u % alignof(decltype(u)) == 0);
  assert((uintptr_t)&f.aligned_field % alignof(decltype(f)) == 0); // The struct should be aligned, but this isn't guaranteed right now
  // see "alignas_keyword_type_aligned_field_fail" test for the case where we check that the struct is also aligned
  assert((uintptr_t)&type_global_var % alignof(decltype(type_global_var)) == 0);
  assert(
    (uintptr_t)&local_var % alignof(decltype(local_var)) != 0); // Should be zero
}

int main()
{
  test_default_fail();
  test_expr_fail();
  test_type_fail();
  return 0;
}
