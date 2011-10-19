// Fig. 6.14: fig06_14.cpp 
// Demonstrating a default constructor for class Time.
#include <iostream>

using std::cout;
using std::endl;

// include definition of class Time from time2.h
#include "time2.h"

int main()
{
   Time t1;               // all arguments defaulted
   Time t2( 2 );          // minute and second defaulted
   Time t3( 21, 34 );     // second defaulted 
   Time t4( 12, 25, 42 ); // all values specified
   Time t5( 27, 74, 99 ); // all bad values specified

   cout << "Constructed with:\n\n"
        << "all default arguments:\n  ";
   t1.printUniversal();  // 00:00:00
   cout << "\n  ";
   t1.printStandard();   // 12:00:00 AM

   cout << "\n\nhour specified; default minute and second:\n  ";
   t2.printUniversal();  // 02:00:00
   cout << "\n  ";
   t2.printStandard();   // 2:00:00 AM

   cout << "\n\nhour and minute specified; default second:\n  ";
   t3.printUniversal();  // 21:34:00
   cout << "\n  ";
   t3.printStandard();   // 9:34:00 PM

   cout << "\n\nhour, minute, and second specified:\n  ";
   t4.printUniversal();  // 12:25:42
   cout << "\n  ";
   t4.printStandard();   // 12:25:42 PM

   cout << "\n\nall invalid values specified:\n  ";
   t5.printUniversal();  // 00:00:00
   cout << "\n  ";
   t5.printStandard();   // 12:00:00 AM
   cout << endl;

   return 0;

} // end main

/**************************************************************************
 * (C) Copyright 1992-2002 by Deitel & Associates, Inc. and Prentice      *
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