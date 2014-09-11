// Fig. 21.33: fig21_33.cpp
// Standard library functions copy_backward, merge,
// unique and reverse.
#include <iostream>
#include <iterator> //Foi adicionada a biblioteca "iterator".
#include <string>
using std::cout;
using std::endl;

#include <algorithm>  // algorithm definitions
#include <vector>     // vector class-template definition

int main()
{
   const int SIZE = 5;
   int a1[ SIZE ] = { 1, 3, 5, 7, 9 };
   int a2[ SIZE ] = { 2, 4, 5, 7, 9 };
   
   std::vector< int > v1( a1, a1 + SIZE );
   std::vector< int > v2( a2, a2 + SIZE );

   std::ostream_iterator< int > output( cout, " " );

   cout << "Vector v1 contains: ";
   std::copy( v1.begin(), v1.end(), output );
   cout << "\nVector v2 contains: ";
   std::copy( v2.begin(), v2.end(), output );

   std::vector< int > results( v1.size() );

   // place elements of v1 into results in reverse order
   std::copy_backward( v1.begin(), v1.end(), results.end() );

   cout << "\n\nAfter copy_backward, results contains: ";
   std::copy( results.begin(), results.end(), output );
   
   std::vector< int > results2( v1.size() + v2.size() );

   // merge elements of v1 and v2 into results2 in sorted order
   std::merge( v1.begin(), v1.end(), v2.begin(), v2.end(),
      results2.begin() );

   cout << "\n\nAfter merge of v1 and v2 results2 contains:\n";
   std::copy( results2.begin(), results2.end(), output );
   
   // eliminate duplicate values from results2
   std::vector< int >::iterator endLocation;
   endLocation = 
      std::unique( results2.begin(), results2.end() );

   cout << "\n\nAfter unique results2 contains:\n";
   std::copy( results2.begin(), endLocation, output );
   
   cout << "\n\nVector v1 after reverse: ";
   
   // reverse elements of v1
   std::reverse( v1.begin(), v1.end() );  

   std::copy( v1.begin(), v1.end(), output );

   cout << endl;

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
