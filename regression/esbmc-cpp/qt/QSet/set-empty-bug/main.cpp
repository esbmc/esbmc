#include <iostream>
#include <QSet>
#include <cassert>
using namespace std;

int main ()
{
  QSet<int> myQSet;
  assert(!myQSet.empty());
  cout << endl;

  return 0;
}
