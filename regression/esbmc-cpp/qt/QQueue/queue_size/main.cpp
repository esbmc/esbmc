// QQueue::size
#include <iostream>
#include <QQueue>
#include <cassert>
using namespace std;

int main ()
{
  QQueue<int> myints;
  cout << "0. size: " << (int) myints.size() << endl;
  assert(myints.size() == 0);

  for (int i=0; i<5; i++) myints.enqueue(i);
  cout << "1. size: " << (int) myints.size() << endl;
  assert(myints.size() == 5);

  myints.dequeue();
  cout << "2. size: " << (int) myints.size() << endl;
  assert(myints.size() == 4);

  return 0;
}
