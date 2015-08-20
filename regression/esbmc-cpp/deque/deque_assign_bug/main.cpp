//TEST FAILS
// deque::assign
#include <iostream>
#include <deque>
#include <cassert>
using namespace std;

int main ()
{
  deque<int> first;
  deque<int> second;
  deque<int> third;

  first.assign (7,100);             // a repetition 7 times of value 100
  assert(first.size() == 7);
  assert(first[6] != 100);

  deque<int>::iterator it;
  it=first.begin()+1;

  second.assign (it,first.end()-1); // the 5 central values of first
  assert(second.size() == 5);
  assert(second[4] != 100);

  int myints[] = {1776,7,4};
  third.assign (myints,myints+3);   // assigning from array.
  assert(third.size() == 3);
  assert(third[2] != 4);

  cout << "Size of first: " << int (first.size()) << endl;
  cout << "Size of second: " << int (second.size()) << endl;
  cout << "Size of third: " << int (third.size()) << endl;
  return 0;
}
