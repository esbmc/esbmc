// QList::pop_back
#include <iostream>
#include <QList>
#include <cassert>
using namespace std;

int main ()
{
  QList<int> myQList;
  int sum (0);
  myQList.push_back (100);
  myQList.push_back (200);
  myQList.push_back (300);
  assert(myQList.back() == 300);
  int n = 3;
  while (!myQList.empty())
  {
    assert(myQList.back() == n*100);
    sum+=myQList.back();
    myQList.pop_back();
    n--;
  }

  cout << "The elements of myQList summed " << sum << endl;

  return 0;
}
