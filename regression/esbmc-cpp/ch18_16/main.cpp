// Fig. 18.26: fig18_26.cpp
// Using strtoul.
#include <iostream>

using std::cout;
using std::endl;

#include <cstdlib>  // strtoul prototype

int main()
{
   unsigned long x;
   const char *string1 = "1234567abc";
   char *remainderPtr;

   x = strtoul( string1, &remainderPtr, 0 );

   cout << "The original string is \"" << string1
        << "\"\nThe converted value is " << x
        << "\nThe remainder of the original string is \""
        << remainderPtr
        << "\"\nThe converted value minus 567 is " 
        << x - 567 << endl;

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