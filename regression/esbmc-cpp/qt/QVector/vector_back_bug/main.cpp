#include <iostream>
#include <QVector>
#include <cassert>
using namespace std;

int main ()
{
  QVector<int> myQVector;

  myQVector.push_back(10);
  int n = 10;
  while (myQVector.back() != 0)
  {
    assert(myQVector.back() != n--);
    myQVector.push_back ( myQVector.back() -1 );
  }

  cout << "myQVector contains:";
  for (QVector<int>::iterator it=myQVector.begin(); it!=myQVector.end() ; ++it)
    cout << " " << *it;

  cout << endl;

  return 0;
}
