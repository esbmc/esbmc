// QVector::pop_front
#include <iostream>
#include <QVector>
#include <cassert>
using namespace std;

int main ()
{
  QVector<int> myQVector;
  myQVector.push_back (100);
  myQVector.push_back (200);
  myQVector.push_back (300);
  assert(myQVector.front() == 100);
  
  int n = 100;
  
  cout << "Popping out the elements in myQVector:";
  while (!myQVector.empty())
  {
    assert(myQVector.front() == n);
    cout << " " << myQVector.front();
    myQVector.pop_front();
    n +=100;
  }
  assert(myQVector.empty());
  cout << "\nFinal size of myQVector is " << int(myQVector.size()) << endl;

  return 0;
}
