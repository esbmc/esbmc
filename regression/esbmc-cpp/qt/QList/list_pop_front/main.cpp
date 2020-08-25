// QList::pop_front
#include <iostream>
#include <QList>
#include <cassert>
using namespace std;

int main ()
{
  QList<int> myQList;
  myQList.push_back (100);
  myQList.push_back (200);
  myQList.push_back (300);
  assert(myQList.front() == 100);
  
  int n = 100;
  
  cout << "Popping out the elements in myQList:";
  while (!myQList.empty())
  {
    assert(myQList.front() == n);
    cout << " " << myQList.front();
    myQList.pop_front();
    n +=100;
  }
  assert(myQList.empty());
  cout << "\nFinal size of myQList is " << int(myQList.size()) << endl;

  return 0;
}
