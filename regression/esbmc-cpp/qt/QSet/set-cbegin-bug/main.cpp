#include <iostream>
#include <cassert>
#include <QSet>
using namespace std;

int main ()
{
  int i;
  QSet<int> myQSet;
  myQSet.insert(75);
  myQSet.insert(23);
  myQSet.insert(65);
  myQSet.insert(42);
  myQSet.insert(13);
  assert(myQSet.size() != 5);
  QSet<int>::const_iterator it;

  cout << "myQSet contains:" << endl;
  for ( it=myQSet.cbegin(), i=0 ; it != myQSet.cend(); it++, i++ ){
    cout << " " << *it;
  }
  cout << endl;

  return 0;
}
