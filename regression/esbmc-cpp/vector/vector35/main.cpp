// inserting into a vector
//#include <iostream>
#include <vector>
#include <cassert>
using namespace std;

int main ()
{
  vector<int> myvector (3,100);
  vector<int>::iterator it;

  it = myvector.begin();
  it = myvector.insert ( it , 200 );

  myvector.insert (it,2,300);

  // "it" no longer valid, get a new one:
  it = myvector.begin();

  vector<int> anothervector (2,400);
  myvector.insert (it+2,anothervector.begin(),anothervector.end());

  int myarray [] = { 501,502,503 };
  myvector.insert (myvector.begin(), myarray, myarray+3);
  if(myvector[2] != 503) assert(0);/*
  //cout << "myvector contains:";
  for (it=myvector.begin(); it<myvector.end(); it++);
  //  cout << " " << *it;
  //cout << endl;
*/
  return 0;
}
