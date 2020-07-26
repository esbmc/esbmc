// QVector::back
#include <iostream>
#include <QVector>
#include <cassert>
using namespace std;

int main ()
{
  QVector<int> myQVector;

  myQVector.push_back(10);
  int n = 10;
    myQVector.reserve(100);
    assert(myQVector.size() == 100);
  while (myQVector.back() != 0)
  {
    assert(myQVector.back() == n--);
    myQVector.push_back ( myQVector.back() - 1 );
  }

  return 0;
}
