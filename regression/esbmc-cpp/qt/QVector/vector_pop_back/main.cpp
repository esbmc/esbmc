// QVector::pop_back
#include <iostream>
#include <QVector>
#include <cassert>
using namespace std;

int main ()
{
  QVector<int> myQVector;
  int sum (0);
  myQVector.push_back (100);
  myQVector.push_back (200);
  myQVector.push_back (300);
  assert(myQVector.back() == 300);
  int n = 3;
  while (!myQVector.empty())
  {
    assert(myQVector.back() == n*100);
    sum+=myQVector.back();
    myQVector.pop_back();
    n--;
  }

  cout << "The elements of myQVector summed " << sum << endl;

  return 0;
}
