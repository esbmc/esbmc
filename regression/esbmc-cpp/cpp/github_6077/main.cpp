#include <assert.h>
int dtor_count;

class A
{
public:
  int num;
  A(int n) : num(n) {}
  ~A() { dtor_count++; }
};

int check()
{
  A a(1);
  return dtor_count; // captured before ~A() runs: must be 0
}

int main()
{
  A a(1);
  assert(dtor_count == 0);
  int r = check();
  assert(r == 0);          // value computed before destructors
  assert(dtor_count == 1); // check()'s local destructed on return
  return 0;
}
