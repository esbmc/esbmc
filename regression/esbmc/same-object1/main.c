#include <assert.h>
#include <stdlib.h>

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
  //assert(!__ESBMC_same_object(p, q));
  //assert(!__ESBMC_same_object(q, p));
  //assert(!__ESBMC_same_object(q, r));

    
  // Same object - should be true
  assert(__ESBMC_same_object(&x, &x));
    
  // Different objects - should be false
  //assert(!__ESBMC_same_object(&x, &y));
  //assert(!__ESBMC_same_object(&y, &x));

  int *s = NULL;
  assert(__ESBMC_same_object(s, NULL));

  return 0;
}
