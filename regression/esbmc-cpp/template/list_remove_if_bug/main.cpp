// list::remove_if
#include <iostream>
#include <list>
#include <cassert>
using namespace std;

// a predicate implemented as a function:
bool single_digit (const int& value) { return (value<10); }

// a predicate implemented as a class:
class is_odd
{
public:
  bool operator() (const int& value) {return (value%2)==1; }
};

int main ()
{
  int myints[]= {15,36,7,17,20,39,4,1};
  list<int> mylist (myints,myints+8);   // 15 36 7 17 20 39 4 1
  list<int>::iterator it;

  mylist.remove_if (single_digit);      // 15 36 17 20 39
  assert(mylist.size() != 5);
  it = mylist.begin();
  advance(it,2);
  assert(*it != 17);

  mylist.remove_if (is_odd());          // 36 20
  assert(mylist.size() != 2);
  it = mylist.begin();
  it++;
  assert(*it != 20);

  cout << "mylist contains:";
  for (list<int>::iterator it=mylist.begin(); it!=mylist.end(); ++it)
    cout << " " << *it;
  cout << endl;

  return 0;
}
