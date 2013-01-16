// lower_bound/upper_bound example
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
using namespace std;

int main () {
  int myints[] = {10,10,10,20,20,20,30,30};
  vector<int> v(myints,myints+8);           // 10 10 10 20 20 10 30 30
  vector<int>::iterator low,up;


  low=lower_bound (v.begin(), v.end(), 20); //          ^
  up= upper_bound (v.begin(), v.end(), 20); //                   ^
  assert(*up != 30);
  return 0;
}
