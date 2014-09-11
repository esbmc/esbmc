// Fig. 21.15: fig21_15.cpp
// Testing Standard Library vector class template 
// element-manipulation functions.
#include <string>

#include <iostream>
using std::cout;
using std::endl;

#include <vector>     // vector class-template definition
#include <algorithm>  // copy algorithm


#include <iterator> // ostream_iterator iterator
#include <stdexcept> // out_of_range exception

int main()
{
   const int SIZE = 6;   
   int array[ SIZE ] = { 1, 2, 3, 4, 5, 6 };

   std::vector< int > integers( array, array + SIZE );
   std::ostream_iterator< int > output( cout, " " );

   cout << "Vector integers contains: ";
   std::copy( integers.begin(), integers.end(), output );

   cout << "\nFirst element of integers: " << integers.front()
        << "\nLast element of integers: " << integers.back();

   integers[ 0 ] = 7;      // set first element to 7

   integers.at( 2 ) = 10;  // set element at position 2 to 10

   // insert 22 as 2nd element
   integers.insert( integers.begin() + 1, 22 );  

   cout << "\n\nContents of vector integers after changes: ";
   std::copy( integers.begin(), integers.end(), output );

   // access out-of-range element
   try {
      integers.at( 100 ) = 777;   

   } // end try

   // catch out_of_range exception
   catch ( std::out_of_range outOfRange ) {
      cout << "\n\nException: " << outOfRange.what();

   } // end catch

   // erase first element
   integers.erase( integers.begin() );
   cout << "\n\nVector integers after erasing first element: ";
   std::copy( integers.begin(), integers.end(), output );

   // erase remaining elements
   integers.erase( integers.begin(), integers.end() );
   cout << "\nAfter erasing all elements, vector integers " 
        << ( integers.empty() ? "is" : "is not" ) << " empty";

   // insert elements from array
   integers.insert( integers.begin(), array, array + SIZE );
   cout << "\n\nContents of vector integers before clear: ";
   std::copy( integers.begin(), integers.end(), output );

   // empty integers; clear calls erase to empty a collection
   integers.clear();  
   cout << "\nAfter clear, vector integers " 
        << ( integers.empty() ? "is" : "is not" ) << " empty";

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
