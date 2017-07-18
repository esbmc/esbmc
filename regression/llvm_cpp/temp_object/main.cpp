#include <assert.h>
int x1 = 0;

struct X 
{ 
  X(int a, float b)  { x1++; };
};

X create_X() 
{
   return X(1, 3.14f); // creates a CXXTemporaryObjectExpr
};

int main(void)
{
  const struct X &x_ref = X(2, 3.14);
  struct X x = create_X();

  assert(x1 == 2);
  return 0;
}
