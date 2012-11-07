// Fig. 21.27: fig21_27.cpp
// Standard library functions equal, 
// mismatch and lexicographical_compare.
#include <iostream>
#include <iterator> //Foi adicionada a biblioteca "iterator".
#include <string>
using std::cout;
using std::endl;

#include <algorithm>  // algorithm definitions
#include <vector>     // vector class-template definition

int main()
{
   const int SIZE = 10;
   int a1[ SIZE ] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
   int a2[ SIZE ] = { 1, 2, 3, 4, 1000, 6, 7, 8, 9, 10 };

   std::vector< int > v1( a1, a1 + SIZE );
   std::vector< int > v2( a1, a1 + SIZE );
   std::vector< int > v3( a2, a2 + SIZE );

   std::ostream_iterator< int > output( cout, " " );

   cout << "Vector v1 contains: ";
   std::copy( v1.begin(), v1.end(), output );
   cout << "\nVector v2 contains: ";
   std::copy( v2.begin(), v2.end(), output );
   cout << "\nVector v3 contains: ";
   std::copy( v3.begin(), v3.end(), output );

   // compare vectors v1 and v2 for equality
   bool result = 
      std::equal( v1.begin(), v1.end(), v2.begin() );

   cout << "\n\nVector v1 " << ( result ? "is" : "is not" ) 
        << " equal to vector v2.\n";

   // compare vectors v1 and v3 for equality
   result = std::equal( v1.begin(), v1.end(), v3.begin() );
   cout << "Vector v1 " << ( result ? "is" : "is not" ) 
        << " equal to vector v3.\n";

   // location represents pair of vector iterators
   std::pair< std::vector< int >::iterator,
              std::vector< int >::iterator > location;

   // check for mismatch between v1 and v3
   location = 
      std::mismatch( v1.begin(), v1.end(), v3.begin() );

   cout << "\nThere is a mismatch between v1 and v3 at "
        << "location " << ( location.first - v1.begin() ) 
        << "\nwhere v1 contains " << *location.first
        << " and v3 contains " << *location.second 
        << "\n\n";

   char c1[ SIZE ] = "HELLO";
   char c2[ SIZE ] = "BYE BYE";

   // perform lexicographical comparison of c1 and c2
   result = std::lexicographical_compare( 
      c1, c1 + SIZE, c2, c2 + SIZE );

   cout << c1 
        << ( result ? " is less than " : 
           " is greater than or equal to " )
        << c2 << endl;

   return 0;

} // end main

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
