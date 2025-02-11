// priority_queue::size
#include <cassert>
#include <cstdlib>
#include <queue>

using namespace std;

int nondet_int();
int N = nondet_int();

int main ()
{
  __ESBMC_assume(N>0);

  priority_queue<int> myints;

  for (int i=0; i<=N; i++) 
    myints.push(i);

  assert(myints.size() == N+1);
  myints.pop();
  assert(myints.size() == N+1);
  return 0;
}
