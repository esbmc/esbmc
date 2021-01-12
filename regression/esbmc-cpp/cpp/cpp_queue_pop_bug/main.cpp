// queue::push/pop
#include <cassert>
#include <queue>
using namespace std;

int nondet_int();
int N = nondet_int();

int main ()
{
  queue<int> myqueue;
  int myint;

  __ESBMC_assume(N>0);

  int i;
  for(i = 0; i < N; i++)
    myqueue.push(i);
   
  while (!myqueue.size())
  {
    assert(myqueue.front() != N-i--);
    myqueue.pop();
  }

  assert(myqueue.size() == 0);

  return 0;
}
