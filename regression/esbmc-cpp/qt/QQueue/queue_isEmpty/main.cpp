// QQueue::isEmpty
#include <iostream>
#include <QQueue>
#include <cassert>
using namespace std;

int main ()
{
  QQueue<int> myQQueue;
  int sum (0);

  for (int i=1;i<=10;i++) myQQueue.enqueue(i);

  while (!myQQueue.isEmpty())
  {
     sum += myQQueue.head();
     myQQueue.dequeue();
  }
  assert(myQQueue.isEmpty());
  cout << "total: " << sum << endl;
  
  return 0;
}
