// list::unique
#include <iostream>
#include <cmath>
#include <list>
#include <cassert>
using namespace std;

// a binary predicate implemented as a function:
bool same_integral_part (double first, double second)
{ return ( int(first)==int(second) ); }

// a binary predicate implemented as a class:
class is_near
{
public:
  bool operator() (double first, double second)
  { return (fabs(first-second)<5.0); }
};

int main ()
{
  double mydoubles[]={ 12.15,  2.72, 73.0,  12.77,  3.14,
                       12.77, 73.35, 72.25, 15.3,  72.25 };
  list<double> mylist (mydoubles,mydoubles+10);
  
  mylist.sort();             //  2.72,  3.14, 12.15, 12.77, 12.77,
                             // 15.3,  72.25, 72.25, 73.0,  73.35

  mylist.unique();           //  2.72,  3.14, 12.15, 12.77
                             // 15.3,  72.25, 73.0,  73.35
  assert(mylist.size() == 8);
  list<double>::iterator it = mylist.begin();
  assert(*it == 2.72);it++;
  assert(*it == 3.14);it++;
  assert(*it != 12.15);it++;
  assert(*it == 12.77);it++;
  assert(*it == 15.3);it++;
  assert(*it != 72.25);it++;
  assert(*it == 73.0);it++;
  assert(*it == 73.35);

  return 0;
}
