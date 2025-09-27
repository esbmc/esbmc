#include <assert.h>
#include <stdlib.h>
#include <stddef.h>

struct SimpleStruct {
  char c;
  int d;
  double e;
};

struct NestedStruct {
  char x;
  struct SimpleStruct inner;
  int z;
};

union TestUnion {
  int i;
  char c;
  double d;
};

struct OuterStruct {
    int data[10];
    struct InnerStruct {
        double x[5];
        char y;
    } inner;
};

int main()
{
  int *p, *q;
  int x = 10, y = -1;

  // Symbolic values: use __ESBMC_symbolic_int to define symbolic variables
  int *symbolic_ptr1, *symbolic_ptr2;

  // Different stack objects → offset 0 for both
  p = &x;
  q = &y;
  assert(__ESBMC_POINTER_OFFSET(p) == __ESBMC_POINTER_OFFSET(q));

  int a[10], b[5];
  
  // Symbolic array indexing and checking offsets

  // Array bases also start at offset 0
  assert(__ESBMC_POINTER_OFFSET(a) == __ESBMC_POINTER_OFFSET(b));

  // Symbolic array indexing → offsets increase correctly
  assert(__ESBMC_POINTER_OFFSET(&a[0]) == 0);
  assert(__ESBMC_POINTER_OFFSET(&a[1]) == sizeof(int));
  assert(__ESBMC_POINTER_OFFSET(&a[9]) == 9 * sizeof(int));

  // Symbolic Pointer arithmetic → consistency
  p = a;
  assert(__ESBMC_POINTER_OFFSET(p + 2) == 2 * sizeof(int));
  assert(__ESBMC_POINTER_OFFSET(p - 1) == -1 * (int)sizeof(int));

  // Symbolic Struct members → different offsets
  struct SimpleStruct s;
  assert(__ESBMC_POINTER_OFFSET(&s.c) == 0);

  // Null pointer → defined as offset 0 in ESBMC
  int *nullp = 0;
  assert(__ESBMC_POINTER_OFFSET(nullp) == 0);

  // All struct members should have correct offsets
  assert(__ESBMC_POINTER_OFFSET(&s.c) == offsetof(struct SimpleStruct, c));
  assert(__ESBMC_POINTER_OFFSET(&s.d) == offsetof(struct SimpleStruct, d));
  assert(__ESBMC_POINTER_OFFSET(&s.e) == offsetof(struct SimpleStruct, e));

  // All union members should have offset 0 relative to union base
  union TestUnion u;
  assert(__ESBMC_POINTER_OFFSET(&u.i) == __ESBMC_POINTER_OFFSET(&u));
  assert(__ESBMC_POINTER_OFFSET(&u.c) == __ESBMC_POINTER_OFFSET(&u));
  assert(__ESBMC_POINTER_OFFSET(&u.d) == __ESBMC_POINTER_OFFSET(&u));

  // Symbolic multi-dimensional Array Indexing
  int matrix[3][4][5];
  int symbolic_idx1 = 2, symbolic_idx2 = 1, symbolic_idx3 = 3;

  // Base array has offset 0
  assert(__ESBMC_POINTER_OFFSET(matrix) == 0);
  assert(__ESBMC_POINTER_OFFSET(&matrix[0]) == 0);
  assert(__ESBMC_POINTER_OFFSET(&matrix[0][0]) == 0);
  assert(__ESBMC_POINTER_OFFSET(&matrix[0][0][0]) == 0);

  // Nested indexing with symbolic offsets
  assert(__ESBMC_POINTER_OFFSET(&matrix[symbolic_idx1][symbolic_idx2][symbolic_idx3]) == 
         (symbolic_idx1 * 4 * 5 + symbolic_idx2 * 5 + symbolic_idx3) * sizeof(int));

  // Pointer subtraction with symbolic offsets
  int arr[20];
  int *ptr1 = &arr[10], *ptr2 = &arr[3];

  // ptr1 - ptr2 should give difference in offsets divided by element size
  // But for pointer_offset, we want the offset from base
  assert(__ESBMC_POINTER_OFFSET(ptr1) == 10 * sizeof(int));
  assert(__ESBMC_POINTER_OFFSET(ptr2) == 3 * sizeof(int));

  // Symbolic offset differences
  int symbolic_offset = 7;
  assert(__ESBMC_POINTER_OFFSET(&arr[symbolic_offset]) == symbolic_offset * sizeof(int));

  // Adding/subtracting zero should not change offset
  int *base_ptr = &arr[5];
  assert(__ESBMC_POINTER_OFFSET(base_ptr + 0) == __ESBMC_POINTER_OFFSET(base_ptr));
  assert(__ESBMC_POINTER_OFFSET(base_ptr - 0) == __ESBMC_POINTER_OFFSET(base_ptr));

  // Chained member access should accumulate offsets correctly
  struct NestedStruct nested;

  // Base struct members
  assert(__ESBMC_POINTER_OFFSET(&nested.x) == offsetof(struct NestedStruct, x));
  assert(__ESBMC_POINTER_OFFSET(&nested.inner) == offsetof(struct NestedStruct, inner));
  assert(__ESBMC_POINTER_OFFSET(&nested.z) == offsetof(struct NestedStruct, z));

  // Nested struct members - should accumulate offsets
  assert(__ESBMC_POINTER_OFFSET(&nested.inner.c) ==
         offsetof(struct NestedStruct, inner) + offsetof(struct SimpleStruct, c));
  assert(__ESBMC_POINTER_OFFSET(&nested.inner.d) ==
         offsetof(struct NestedStruct, inner) + offsetof(struct SimpleStruct, d));
  assert(__ESBMC_POINTER_OFFSET(&nested.inner.e) ==
         offsetof(struct NestedStruct, inner) + offsetof(struct SimpleStruct, e));

  // Complex arithmetic expressions should be simplified
  int *complex_ptr = arr + 5;

  // (ptr + n) + m should equal ptr + (n + m)
  assert(__ESBMC_POINTER_OFFSET((arr + 3) + 2) == __ESBMC_POINTER_OFFSET(arr + 5));
  assert(__ESBMC_POINTER_OFFSET((arr + 7) - 2) == __ESBMC_POINTER_OFFSET(arr + 5));

  // More complex: ((ptr + a) - b) + c should simplify to ptr + (a - b + c)
  assert(__ESBMC_POINTER_OFFSET(((arr + 10) - 3) + 2) == __ESBMC_POINTER_OFFSET(arr + 9));

  // Symbolic array indices in different data types
  char char_array[100];
  double double_array[50];

  assert(__ESBMC_POINTER_OFFSET(char_array) == 0);
  assert(__ESBMC_POINTER_OFFSET(double_array) == 0);
  
  // Arrays inside structs
  struct ArrayStruct {
    int prefix;
    int data[10];
    int suffix;
  } array_struct;

  assert(__ESBMC_POINTER_OFFSET(&array_struct.data[0]) ==
         offsetof(struct ArrayStruct, data));
  assert(__ESBMC_POINTER_OFFSET(&array_struct.data[5]) ==
         offsetof(struct ArrayStruct, data) + 5 * sizeof(int));
  // Pointer casts should preserve offsets
  char *char_ptr = (char*)(&arr[4]);
  assert(__ESBMC_POINTER_OFFSET(char_ptr) == 4 * sizeof(int));

  // Very large array indices (within bounds)
  int large_array[1000];
  assert(__ESBMC_POINTER_OFFSET(&large_array[999]) == 999 * sizeof(int));

  // Negative pointer arithmetic with symbolic values
  int *mid_ptr = &large_array[500];
  int symbolic_offset_neg = -100;

  assert(__ESBMC_POINTER_OFFSET(mid_ptr - symbolic_offset_neg) == 600 * sizeof(int));
 
  // Complex typecasting scenarios with pointer arithmetic
  int arr2[10];
  int symbolic_index = 7;
  assert(__ESBMC_POINTER_OFFSET(&arr2[symbolic_index]) == symbolic_index * sizeof(int));

  // Handling NULL Pointer
  assert(__ESBMC_POINTER_OFFSET(nullp) == 0);

  // Symbolic Pointer Arithmetic on Struct with Nested Arrays
  struct OuterStruct outer;
  symbolic_index = 7;
  assert(__ESBMC_POINTER_OFFSET(&outer.data[symbolic_index]) == symbolic_index * sizeof(int));
  assert(__ESBMC_POINTER_OFFSET(&outer.inner.x[symbolic_index]) != symbolic_index * sizeof(double));
 
  return 0;
}

