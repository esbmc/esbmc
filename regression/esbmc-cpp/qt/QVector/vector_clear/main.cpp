#include <iostream>
#include <QVector>
#include <cassert>
using namespace std;

int main ()
{
  QVector<int> myQVector;
  QVector<int>::iterator it;

  myQVector.push_back (100);
  myQVector.push_back (200);
  myQVector.push_back (300);

  cout << "myQVector contains:";
  for (it=myQVector.begin(); it!=myQVector.end(); ++it)
    cout << " " << *it;
  assert(myQVector.size() == 3);
  myQVector.clear();
  assert(myQVector.size() == 0);
  myQVector.push_back (1101);
  myQVector.push_back (2202);
  assert(myQVector.size() == 2);
  cout << "\nmyQVector contains:";
  for (it=myQVector.begin(); it!=myQVector.end(); ++it)
    cout << " " << *it;

  cout << endl;

  return 0;
}
