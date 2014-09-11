// Fig. 21.26: fig21_26.cpp
// Standard library algorithms fill, fill_n, generate
// and generate_n.
#include <iostream>
#include <string>
using std::cout;
using std::endl;

#include <algorithm>  // algorithm definitions
#include <vector>     // vector class-template definition

char nextLetter();    // prototype

int main()
{
   std::vector< char > chars( 10 );
   std::ostreambuf_iterator< char > output( cout );

   // fill chars with 5s
   std::fill( chars.begin(), chars.end(), '5' );

   cout << "Vector chars after filling with 5s:\n";
   std::copy( chars.begin(), chars.end(), output );

   // fill first five elements of chars with As
   std::fill_n( chars.begin(), 5, 'A' );

   cout << "\n\nVector chars after filling five elements"
        << " with As:\n";
   std::copy( chars.begin(), chars.end(), output );

   // generate values for all elements of chars with nextLetter
   std::generate( chars.begin(), chars.end(), nextLetter );

   cout << "\n\nVector chars after generating letters A-J:\n";
   std::copy( chars.begin(), chars.end(), output );

   // generate values for first five elements of chars 
   // with nextLetter
   std::generate_n( chars.begin(), 5, nextLetter );

   cout << "\n\nVector chars after generating K-O for the"
        << " first five elements:\n";
   std::copy( chars.begin(), chars.end(), output );

   cout << endl;

   return 0;

} // end main

// returns next letter in the alphabet (starts with A)
char nextLetter()
{
   static char letter = 'A';
   return letter++;

} // end function nextLetter

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
