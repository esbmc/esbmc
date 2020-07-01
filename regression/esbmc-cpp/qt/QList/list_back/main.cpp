// QList::back
#include <iostream>
#include <QList>
#include <cassert>
using namespace std;

int main ()
{
  QList<int> myQList;

  myQList.push_back(10);
  int n = 10;
  while (myQList.back() != 0)
  {
    assert(myQList.back() == n--);
    myQList.push_back ( myQList.back() -1 );
  }

  cout << "myQList contains:";
  for (QList<int>::iterator it=myQList.begin(); it!=myQList.end() ; ++it)
    cout << " " << *it;

  cout << endl;

  return 0;
}
