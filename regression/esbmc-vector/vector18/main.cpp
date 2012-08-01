// vector_at.cpp
// compile with: /EHsc
#include <vector>
#include <iostream>
using namespace std;

int main( )
{
   vector <int> v1;
   
   v1.push_back( 10 );
   v1.push_back( 20 );

   const int &i = v1.at( 0 );
   int &j = v1.at( 1 );
   cout << "The first element is " << i << endl;
   cout << "The second element is " << j << endl;
}
