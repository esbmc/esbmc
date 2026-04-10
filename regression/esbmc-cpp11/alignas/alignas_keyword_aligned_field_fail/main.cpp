#include <assert.h>
#include <stdint.h>
#include <stddef.h>

// alignas(max_align_t) on field types (emulate default align behaviour)

// Struct with aligned field
struct default_field_aligned
{
  char prefix;
  alignas(max_align_t) int aligned_field;
  char suffix;
};

// alignas(expr) on field types

// Struct with aligned field
struct expr_field_aligned
{
  char prefix;
  alignas(8) int aligned_field;
  char suffix;
};

// alignas(type) on field types

// Struct with aligned field
struct type_field_aligned
{
  char prefix;
  alignas(float) int aligned_field;
  char suffix;
};

// Test: alignas(max_align_t) on field (emulate default align behaviour)
void test_default_aligned_field_fail()
{
  struct default_field_aligned f;

  f.aligned_field = 200;
  assert(f.aligned_field == 200);
  assert((uintptr_t)&f % alignof(decltype(f)) != 0); // Should be 0
}

// Test: alignas(expr) on field
void test_expr_aligned_field_fail()
{
  struct expr_field_aligned f;

  f.aligned_field = 200;

  assert(f.aligned_field == 200);
  assert((uintptr_t)&f % alignof(decltype(f)) != 0); // Should be 0
}

// Test: alignas(type) on field
void test_type_aligned_field_fail()
{
  struct type_field_aligned f;

  f.aligned_field = 200;

  assert(f.aligned_field == 200);
  assert((uintptr_t)&f % alignof(decltype(f)) != 0); // Should be 0
}

int main()
{
  test_default_aligned_field_fail();
  test_expr_aligned_field_fail();
  test_type_aligned_field_fail();
  return 0;
}
