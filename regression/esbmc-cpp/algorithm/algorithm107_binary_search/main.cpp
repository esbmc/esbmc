// binary_search example
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
using namespace std;

bool myfunction (int i,int j) { return (i<j); }

int main () {
  int myints[] = {1,2,3,4,5,6,7,8,9,10,11,12};
  int myints1[] = {1,2,3,4,5,6,7,8,9,10,11,12};
  vector<int> v(myints,myints+11);                         // 1 2 3 4 5 4 3 2 1
  vector<int>::iterator it;

  // using default comparison:
  v.assign(myints1,myints1+11);                      //  1 1 2 2 3 3 4 4 5

  cout << "looking for a 3... ";/*
  //if (binary_search (v.begin(), v.end(), 3))
    //cout << "found!\n"; else cout << "not found.\n";
  assert(!binary_search (v.begin(), v.end(), 3));

  // using myfunction as comp:
  sort (v.begin(), v.end(), myfunction);
  for(it = v.begin();it < v.end();it++)
  	cout << " " << *it << " " ;
  */
  assert(!binary_search (v.begin(), v.end(), 6, myfunction));
/*  cout << "looking for a 6... ";
  if (binary_search (v.begin(), v.end(), 6, myfunction))
    cout << "found!\n"; else cout << "not found.\n";
*/
  return 0;
}
