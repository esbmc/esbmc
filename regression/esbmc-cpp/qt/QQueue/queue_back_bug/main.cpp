// QQueue::head
#include <iostream>
#include <QQueue>
#include <cassert>
using namespace std;

int main ()
{
  QQueue<int> myQQueue;

  myQQueue.enqueue(12);
  myQQueue.enqueue(75);   // this is now the back
  assert(myQQueue.back() != 75);
  myQQueue.back() -= myQQueue.head();
  assert(myQQueue.back() != 63);
  cout << "myQQueue.back() is now " << myQQueue.back() << endl;

  return 0;
}
