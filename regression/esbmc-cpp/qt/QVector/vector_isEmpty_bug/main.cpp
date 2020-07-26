// QVector::back
#include <iostream>
#include <QVector>
#include <cassert>
using namespace std;

int main ()
{
  QVector<int> myQVector;
    assert(myQVector.isEmpty());
    myQVector.push_back(10);
    myQVector.push_back(10);
    assert(myQVector.isEmpty());
  return 0;
}
