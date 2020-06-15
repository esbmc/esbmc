// clearing QLists
#include <iostream>
#include <QList>
#include <cassert>
using namespace std;

int main ()
{
  QList<int> myQList;
  QList<int>::iterator it;

  myQList.push_back (100);
  myQList.push_back (200);
  myQList.push_back (300);

  cout << "myQList contains:";
  for (it=myQList.begin(); it!=myQList.end(); ++it)
    cout << " " << *it;
  assert(myQList.size() == 3);
  myQList.clear();
  assert(myQList.size() == 0);
  myQList.push_back (1101);
  myQList.push_back (2202);
  assert(myQList.size() == 2);
  cout << "\nmyQList contains:";
  for (it=myQList.begin(); it!=myQList.end(); ++it)
    cout << " " << *it;

  cout << endl;

  return 0;
}
