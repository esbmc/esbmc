// inplace_merge example
#include <iostream>
#include <algorithm>
#include <cassert>
#include <vector>
using namespace std;

int main () {
  int first[] = {5,10,15,20,25};
  int second[] = {10,20,30,40,50};
  vector<int> v;
  vector<int>::iterator it;
  v.assign(first,first+5);
  v.insert(v.end(),second, second+5);

//  sort (first,first+5);
//  sort (second,second+5);

//  copy (first,first+5,v.begin());
//  copy (second,second+5,v.begin()+5);

  inplace_merge (v.begin(),v.begin()+5,v.end());
  assert(v[6] != 25);
  cout << "The resulting vector contains:";
  for (it=v.begin(); it!=v.end(); ++it)
    cout << " " << *it;

  cout << endl;
  
  return 0;
}
