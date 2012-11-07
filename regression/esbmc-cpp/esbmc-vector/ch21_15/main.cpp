// Fig. 21.28: fig21_28.cpp
// Standard library functions remove, remove_if,
// remove_copy and remove_copy_if.
#include <iostream>
#include <iterator> //Foi adicionada a biblioteca "iterator".
#include <string>
using std::cout;
using std::endl;

#include <algorithm>   // algorithm definitions
#include <vector>      // vector class-template definition

bool greater9( int );  // prototype

int main()
{ 
   const int SIZE = 10;
   int a[ SIZE ] = { 10, 2, 10, 4, 16, 6, 14, 8, 12, 10 };

   std::ostream_iterator< int > output( cout, " " );

   std::vector< int > v( a, a + SIZE );  
   std::vector< int >::iterator newLastElement;
   
   cout << "Vector v before removing all 10s:\n   ";
   std::copy( v.begin(), v.end(), output );

   // remove 10 from v
   newLastElement = std::remove( v.begin(), v.end(), 10 );

   cout << "\nVector v after removing all 10s:\n   ";
   std::copy( v.begin(), newLastElement, output );

   std::vector< int > v2( a, a + SIZE );
   std::vector< int > c( SIZE, 0 );

   cout << "\n\nVector v2 before removing all 10s "
        << "and copying:\n   ";
   std::copy( v2.begin(), v2.end(), output );

   // copy from v2 to c, removing 10s in the process
   std::remove_copy( v2.begin(), v2.end(), c.begin(), 10 );

   cout << "\nVector c after removing all 10s from v2:\n   ";
   std::copy( c.begin(), c.end(), output );

   std::vector< int > v3( a, a + SIZE );

   cout << "\n\nVector v3 before removing all elements"
        << "\ngreater than 9:\n   ";
   std::copy( v3.begin(), v3.end(), output );

   // remove elements greater than 9 from v3
   newLastElement = 
      std::remove_if( v3.begin(), v3.end(), greater9 );

   cout << "\nVector v3 after removing all elements"
        << "\ngreater than 9:\n   ";
   std::copy( v3.begin(), newLastElement, output );

   std::vector< int > v4( a, a + SIZE );
   std::vector< int > c2( SIZE, 0 );

   cout << "\n\nVector v4 before removing all elements"
        << "\ngreater than 9 and copying:\n   ";
   std::copy( v4.begin(), v4.end(), output );
   
   // copy elements from v4 to c2, removing elements greater
   // than 9 in the process
   std::remove_copy_if(
      v4.begin(), v4.end(), c2.begin(), greater9 );

   cout << "\nVector c2 after removing all elements"
        << "\ngreater than 9 from v4:\n   ";
   std::copy( c2.begin(), c2.end(), output );

   cout << endl;

   return 0;

} // end main

// determine whether argument is greater than 9
bool greater9( int x )
{
   return x > 9;

} // end greater9

/**************************************************************************
 * (C) Copyright 1992-2003 by Deitel & Associates, Inc. and Prentice      *
 * Hall. All Rights Reserved.                                             *
 *                                                                        *
 * DISCLAIMER: The authors and publisher of this book have used their     *
 * best efforts in preparing the book. These efforts include the          *
 * development, research, and testing of the theories and programs        *
 * to determine their effectiveness. The authors and publisher make       *
 * no warranty of any kind, expressed or implied, with regard to these    *
 * programs or to the documentation contained in these books. The authors *
 * and publisher shall not be liable in any event for incidental or       *
 * consequential damages in connection with, or arising out of, the       *
 * furnishing, performance, or use of these programs.                     *
 *************************************************************************/
