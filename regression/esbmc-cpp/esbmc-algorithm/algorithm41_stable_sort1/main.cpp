// stable_sort example
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
using namespace std;

bool compare_as_ints (double i,double j)
{
  return (int(i)<int(j));
}

//using default comparison: 1.32 1.41 1.62 1.73
//using 'compare_as_ints' : 1.41 1.73 1.32 1.62

int main () {
  double mydoubles[] = {1.41, 1.73, 1.32, 1.62};

  vector<double> myvector;
  vector<double>::iterator it;

  myvector.assign(mydoubles,mydoubles+4);

  cout << "using default comparison:";
//  stable_sort (myvector.begin(), myvector.end());
  
//  for (it=myvector.begin(); it!=myvector.end(); ++it)
//    cout << " " << *it;

//  myvector.assign(mydoubles,mydoubles+4);

  cout << "\nusing 'compare_as_ints' :";
  stable_sort (myvector.begin(), myvector.end(), compare_as_ints);
  assert(myvector[0] == 1.41);
  assert(myvector[1] == 1.73);
  assert(myvector[2] == 1.32);
  assert(myvector[3] == 1.62);
//  for (it=myvector.begin(); it!=myvector.end(); ++it)
//    cout << " " << *it;

  cout << endl;

  return 0;
}
