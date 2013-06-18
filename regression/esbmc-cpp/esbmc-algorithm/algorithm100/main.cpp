// stable_sort example
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
using namespace std;

bool compare_as_ints (double i,double j)
{
  return ((int) i< (int) j);
}

int main () {
  double mydoubles[] = {1.41, 4.67, 1.73, 1.32, 1.62};

  vector<double> myvector;
  vector<double>::iterator it;

  myvector.assign(mydoubles,mydoubles+5);
/*
  cout << "using default comparison:";
  stable_sort (myvector.begin(), myvector.end());
  for (it=myvector.begin(); it!=myvector.end(); ++it)
    cout << " " << *it;
  assert(myvector[0] == 1.32);
  myvector.assign(mydoubles,mydoubles+5);
*/
  cout << "\nusing 'compare_as_ints' :";
  stable_sort (myvector.begin(), myvector.end(), compare_as_ints);
  assert(myvector[0] != 1.41);
  for (it=myvector.begin(); it!=myvector.end(); ++it)
    cout << " " << *it;

  cout << endl;

  return 0;
}
