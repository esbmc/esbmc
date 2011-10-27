// Fig. 16.10: fig16_10.cpp
// Using an iterator to output a string.
#include <iostream>

using std::cout;
using std::endl;

#include <string>

using std::string;

int main()
{
   string string1( "Testing iterators" );
   string::const_iterator iterator1 = string1.begin();

   cout << "string1 = " << string1
        << "\n(Using iterator iterator1) string1 is: ";

   // iterate through string
   while ( iterator1 != string1.end() ) {
      cout << *iterator1;  // dereference iterator to get char
      ++iterator1;  // advance iterator to next char
   } // end while

   cout << endl;
   return 0;

}  // end main

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
