// stack::top
#include <cassert>
#include <stack>
using namespace std;

int nondet_int();
int N = nondet_int();

int main ()
{
  stack<int> mystack;

  __ESBMC_assume(N>0);

  __ESBMC_loop_invariant(i == 0 || mystack.top() == i - 1);
  for(int i=0; i <= N; ++i)
    mystack.push(i);

  assert(mystack.top() != N);
  return 0;
}
