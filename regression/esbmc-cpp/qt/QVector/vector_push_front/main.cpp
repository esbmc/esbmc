// QVector::push_front
#include <iostream>
#include <QVector>
#include <cassert>
using namespace std;

int main ()
{
  QVector<int> myQVector;         // two ints with a value of 100
  myQVector.push_back(100);
  myQVector.push_back(100);
  assert(myQVector.front() == 100);
  myQVector.push_front (200);
  assert(myQVector.front() == 200);
  myQVector.push_front (300);
  assert(myQVector.front() == 300);

  cout << "myQVector contains:";
  for (QVector<int>::iterator it=myQVector.begin(); it!=myQVector.end(); ++it)
    cout << " " << *it;

  cout << endl;
  return 0;
}
