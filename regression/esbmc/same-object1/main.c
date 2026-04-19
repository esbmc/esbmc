#include <assert.h>
#include <stdlib.h>

struct Point {
    int x, y;
};

struct Container {
    int data[3];
    int value;
};

void test_function_params(int *a, int *b, int *c) 
{
  // a and c point to same object, b points to different object
  assert(__ESBMC_same_object(a, c));
  assert(!__ESBMC_same_object(a, b));
  assert(!__ESBMC_same_object(b, c));
}

int main()
{
  int *p, *q, *r;
  int x = 10, y = 11;
  p = &x;
  q = &y;
  r = &x;
    
  // Same pointer to same object - should be true
  assert(__ESBMC_same_object(p, p));
    
  // Different pointers to same object - should be true  
  assert(__ESBMC_same_object(p, r));
  assert(__ESBMC_same_object(r, p));
    
  // Different pointers to different objects - should be false
  assert(!__ESBMC_same_object(p, q));
  assert(!__ESBMC_same_object(q, p));
  assert(!__ESBMC_same_object(q, r));

    
  // Same object - should be true
  assert(__ESBMC_same_object(&x, &x));
    
  // Different objects - should be false
  assert(!__ESBMC_same_object(&x, &y));
  assert(!__ESBMC_same_object(&y, &x));

  int *s = NULL;
  assert(__ESBMC_same_object(s, NULL));

  int arr[5] = {1, 2, 3, 4, 5};
  int other[3] = {6, 7, 8};
    
  // Same array element - should be true
  assert(__ESBMC_same_object(&arr[0], &arr[0]));
  assert(__ESBMC_same_object(&arr[2], &arr[2]));
    
  // Different elements of same array - should be true
  assert(__ESBMC_same_object(&arr[0], &arr[1]));
  assert(__ESBMC_same_object(&arr[1], &arr[3]));
    
  // Elements from different arrays - should be false
  assert(!__ESBMC_same_object(&arr[0], &other[0]));
  assert(!__ESBMC_same_object(&arr[2], &other[1]));

  struct Point p1 = {1, 2};
  struct Point p2 = {3, 4};
    
  // Same member of same struct - should be true
  assert(__ESBMC_same_object(&p1.x, &p1.x));
  assert(__ESBMC_same_object(&p1.y, &p1.y));
    
  // Different members of same struct - should be true
  assert(__ESBMC_same_object(&p1.x, &p1.y));
  assert(__ESBMC_same_object(&p1.y, &p1.x));
    
  // Same member of different structs - should be false
  assert(!__ESBMC_same_object(&p1.x, &p2.x));
  assert(!__ESBMC_same_object(&p1.y, &p2.y));
    
  // Different members of different structs - should be false
  assert(!__ESBMC_same_object(&p1.x, &p2.y));
  assert(!__ESBMC_same_object(&p2.y, &p1.x));

  x = 42;
  char *cp;
  int *ip = &x;
  void *vp = &x;
    
  // Same object through different pointer types - should be true
  assert(__ESBMC_same_object(ip, (int*)vp));
  assert(__ESBMC_same_object(vp, (void*)ip));

  p = NULL;
  q = NULL;
  x = 10;
  r = &x;
    
  // Both NULL - should be true
  assert(__ESBMC_same_object(p, q));
  assert(__ESBMC_same_object(NULL, NULL));
    
  // One NULL, one not - should be false
  assert(!__ESBMC_same_object(p, r));
  assert(!__ESBMC_same_object(r, p));
  assert(!__ESBMC_same_object(NULL, r));
  assert(!__ESBMC_same_object(r, NULL));

  struct Container c1 = {{1, 2, 3}, 42};
  struct Container c2 = {{4, 5, 6}, 84};
    
  // Same complex expression - should be true
  assert(__ESBMC_same_object(&c1.data[0], &c1.data[0]));
  assert(__ESBMC_same_object(&c1.value, &c1.value));
    
  // Different elements in same container - should be true
  assert(__ESBMC_same_object(&c1.data[0], &c1.data[1]));
  assert(__ESBMC_same_object(&c1.data[0], &c1.value));
  assert(__ESBMC_same_object(&c1.value, &c1.data[0]));
    
  // Same type of element in different containers - should be false
  assert(!__ESBMC_same_object(&c1.data[0], &c2.data[0]));
  assert(!__ESBMC_same_object(&c1.value, &c2.value));

  // Test function parameters
  x = 1; 
  y = 2;
  test_function_params(&x, &y, &x);

  return 0;
}
