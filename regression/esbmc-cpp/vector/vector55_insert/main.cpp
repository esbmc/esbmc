// inserting into a vector
//#include <iostream>
#include <vector>
#include <cassert>
using namespace std;

int main ()
{
  vector<int> myvector;
  vector<int>::iterator it;

  myvector.push_back(300);
  myvector.push_back(200);
  myvector.push_back(100);

  it = myvector.begin();

  vector<int> anothervector;
  anothervector.push_back(1);
  anothervector.push_back(2);
  myvector.insert (it+2,anothervector.begin(),anothervector.end());
  assert(myvector[3]!=2);

  int myarray [] = { 501,502,503 };
  myvector.insert (myvector.begin(), myarray, myarray+3);
  assert(myvector[2] != 503); 

  return 0;
}
