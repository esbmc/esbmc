// swap lists
#include <iostream>
#include <list>
#include <cassert>
using namespace std;

int main ()
{
  list<int> first (3,100);   // three ints with a value of 100
  list<int> second (5,200);  // five ints with a value of 200
  list<int> first1(first);
  list<int> second1(second);
  list<int>::iterator it;

  first.swap(second);
  assert(first != second1);
  assert(second != first1);
  cout << "first contains:";
  for (it=first.begin(); it!=first.end(); it++) cout << " " << *it;

  cout << "\nsecond contains:";
  for (it=second.begin(); it!=second.end(); it++) cout << " " << *it;

  cout << endl;

  return 0;
}
