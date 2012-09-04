// stable_sort example
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

bool compare_as_ints (double i,double j)
{
  return (int(i)<int(j));
}

int main () {
  double mydoubles[] = {3.14, 1.41, 2.72, 4.67, 1.73, 1.32, 1.62, 2.58};

  vector<double> myvector;
  vector<double>::iterator it;

  myvector.assign(mydoubles,mydoubles+8);

  cout << "using default comparison:";
  stable_sort (myvector.begin(), myvector.end());
  for (it=myvector.begin(); it!=myvector.end(); ++it)
    cout << " " << *it;

  myvector.assign(mydoubles,mydoubles+8);

  cout << "\nusing 'compare_as_ints' :";
  stable_sort (myvector.begin(), myvector.end(), compare_as_ints);
  for (it=myvector.begin(); it!=myvector.end(); ++it)
    cout << " " << *it;

  cout << endl;

  return 0;
}
