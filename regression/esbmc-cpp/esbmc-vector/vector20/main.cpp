#include <vector>
#include <iostream>
using namespace std;   

int main( )
{
   vector <int> v1;

   v1.push_back( 1 );
   cout << "Current capacity of v1 = " 
      << v1.capacity( ) << endl;
   v1.reserve( 20 );
   cout << "Current capacity of v1 = " 
      << v1.capacity( ) << endl;
}
