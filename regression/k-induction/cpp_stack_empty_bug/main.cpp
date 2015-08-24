// stack::size
#include <cassert>
#include <stack>
using namespace std;

int nondet_int();
int N = nondet_int();

int main ()
{
  stack<int> myints;
  assert(myints.size() == 0);

  __ESBMC_assume(N>0);

  for (int i=0; i<N; i++) 
    myints.push(i);

  assert(myints.size() != N);
  
  return 0;
}
