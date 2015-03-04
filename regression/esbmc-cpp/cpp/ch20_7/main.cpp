// Fig. 20.9: fig20_09.cpp
// Using an anonymous union.
#include <iostream>

using std::cout;
using std::endl;

int main()
{
   // declare an anonymous union
   // members integer1, double1 and charPtr share the same space
   union {
      int integer1;
      double double1;
      char *charPtr;

   };  // end anonymous union

   // declare local variables
   int integer2 = 1;
   double double2 = 3.3;
   char *char2Ptr = "Anonymous";

   // assign value to each union member
   // successively and print each
   cout << integer2 << ' ';
   integer1 = 2;
   cout << integer1 << endl;

   cout << double2 << ' ';
   double1 = 4.4;
   cout << double1 << endl;

   cout << char2Ptr << ' ';
   charPtr = "union";
   cout << charPtr << endl;

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
