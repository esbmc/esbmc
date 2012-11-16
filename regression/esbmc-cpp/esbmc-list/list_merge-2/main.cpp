// list::merge
#include <iostream>
#include <list>
#include <cassert>
using namespace std;

// this compares equal two doubles if
//  their interger equivalents are equal
bool mycomparison (double first, double second)
{ return ( int(first)<int(second) ); }

int main ()
{
  list<double> first, second;
  double mydoubles[] = {1.4, 2.2, 2.9, 3.1, 3.7, 7.1};
  double md2[] = {1.4, 2.2, 2.9, 2.1, 3.1, 3.7, 7.1};
  list<double> merged; 
  merged.assign(mydoubles,mydoubles+6);
  
  first.push_back (3.1);
  first.push_back (2.2);
  first.push_back (2.9);

  second.push_back (3.7);
  second.push_back (7.1);
  second.push_back (1.4);

  first.sort();
  second.sort();

  first.merge(second);
  assert(first == merged);
  
  second.push_back (2.1);
  merged.assign(md2,md2+7);
  first.merge(second,mycomparison);
  assert(merged == first);

  cout << "first contains:";
  for (list<double>::iterator it=first.begin(); it!=first.end(); ++it)
    cout << " " << *it;
  cout << endl;

  return 0;
}
