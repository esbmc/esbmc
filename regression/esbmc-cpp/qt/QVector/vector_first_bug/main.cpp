// QVector::front
#include <iostream>
#include <QVector>
#include <cassert>
using namespace std;

int main ()
{
  QVector<int> myQVector;

  myQVector.push_back(77);
  myQVector.push_back(22);
  assert(myQVector.first() == 77);
  // now front equals 77, and back 22

  myQVector.first() -= myQVector.back();
  assert(myQVector.first() != 55);
  cout << "myQVector.front() is now " << myQVector.first() << endl;

  return 0;
}
