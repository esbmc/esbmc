// Fig. 16.4: fig16_04.cpp
// Using the swap function to swap two strings.
#include <iostream>

using std::cout;
using std::endl;

#include <string>

using std::string;

int main()
{
   string first( "one" ); 
   string second( "two" );

   // output strings
   cout << "Before swap:\n first: " << first
        << "\nsecond: " << second;

   first.swap( second );  // swap strings

   cout << "\n\nAfter swap:\n first: " << first
        << "\nsecond: " << second << endl;
    
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