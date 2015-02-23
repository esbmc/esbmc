// basic_istream_gcount.cpp
// compile with: /EHsc
#include <iostream>
#include <cassert>
using namespace std;

int main( ) 
{
   cout << "Type the letter 'a': ";

   ws( cin );
   char c[10];

   cin.get( &c[0],9 );
   cout << c << endl;
   assert((int)cin.gcount() >= 0 || (int)cin.gcount() < 10);
   cout << cin.gcount( ) << endl;
}
