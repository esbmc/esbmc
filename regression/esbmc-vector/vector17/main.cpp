// vector_assign.cpp
// compile with: /EHsc
#include <vector>
#include <iostream>
#include <cassert>
using namespace std;
int main( )
{
   
   vector<int> v1, v2, v3;
   vector<int>::iterator iter;

   v1.push_back(10);
   v1.push_back(20);
   v1.push_back(30);
   v1.push_back(40);
   v1.push_back(50);

   cout << "v1 = " ;
   for (iter = v1.begin(); iter != v1.end(); iter++)
      cout << *iter << " ";
   cout << endl;

   v2.assign(v1.begin(), v1.end());
   cout << "v2 = ";
   for (iter = v2.begin(); iter != v2.end(); iter++)
      cout << *iter << " ";
   cout << endl;

   v3.assign(7, 4) ;
	assert(v3[6] != 4);
   cout << "v3 = ";
   for (iter = v3.begin(); iter != v3.end(); iter++)
      cout << *iter << " ";
   cout << endl;
}
