// transform algorithm example
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

int op_increase (int i) { return ++i; }
int op_sum (int i, int j) { return i+j; }

int main () {
  vector<int> first;
  vector<int> second;
  vector<int>::iterator it;

  // set some values:
  for (int i=1; i<6; i++) first.push_back (i*10); //  first: 10 20 30 40 50

  second.resize(first.size());     // allocate space
  transform (first.begin(), first.end(), second.begin(), op_increase);
                                                  // second: 11 21 31 41 51

  transform (first.begin(), first.end(), second.begin(), first.begin(), op_sum);
                                                  //  first: 21 41 61 81 101

  cout << "first contains:";
  for (it=first.begin(); it!=first.end(); ++it)
    cout << " " << *it;

  cout << endl;
  return 0;
}
