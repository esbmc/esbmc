// vector assign
#include <iostream>
#include <vector>
#include <cassert>
using namespace std;

int main ()
{
  vector<int> first;
  vector<int> second;
  vector<int> third;

  first.assign (7,100);             // a repetition 7 times of value 100

  vector<int>::iterator it;
  it=first.begin()+1;

  second.assign (it,first.end()-1); // the 5 central values of first

  int myints[] = {1776,7,4};
  third.assign (myints,myints+3);   // assigning from array.

  cout << "Size of first: " << int (first.size()) << endl;
  assert(first.size() == 7);
  cout << "Size of second: " << int (second.size()) << endl;
  assert(second.size() == 5);
  cout << "Size of third: " << int (third.size()) << endl;
  assert(third.size() == 3);
  return 0;
}
