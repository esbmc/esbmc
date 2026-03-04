#include <assert.h>
#include <stdint.h>

// __attribute__((aligned)) (default value) on field types

// Struct with aligned field
struct default_field_aligned
{
  char prefix;
  int __attribute__((aligned)) aligned_field;
  char suffix;
};

// __attribute__((aligned(expr))) on field types

// Struct with aligned field
struct expr_field_aligned
{
  char prefix;
  int __attribute__((aligned(8))) aligned_field;
  char suffix;
};

// Test: __attribute__((aligned)) (default value) on field
void test_default_aligned_field_fail()
{
  struct default_field_aligned f;

  f.aligned_field = 200;

  assert(f.aligned_field == 200);
  assert((uintptr_t) &f % _Alignof(typeof(f)) != 0); // Should be 0
}

// Test: __attribute__((aligned(expr))) on field
void test_expr_aligned_field_fail()
{
  struct expr_field_aligned f;

  f.aligned_field = 200;

  assert(f.aligned_field == 200);
  assert((uintptr_t) &f % _Alignof(typeof(f)) != 0); // Should be 0
}

int main()
{
  test_default_aligned_field_fail();
  test_expr_aligned_field_fail();
  return 0;
}
