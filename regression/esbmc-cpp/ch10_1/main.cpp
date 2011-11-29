// Fig. 10.5: fig10_05.cpp
// Aiming base-class and derived-class pointers at base-class
// and derived-class objects, respectively.
#include <iostream>

using std::cout;
using std::endl;
using std::fixed;

#include <iomanip>

using std::setprecision;

#include "point.h"   // Point class definition
#include "circle.h"  // Circle class definition

int main()
{
   Point point( 30, 50 );
   Point *pointPtr = 0;    // base-class pointer

   Circle circle( 120, 89, 2.7 );
   Circle *circlePtr = 0;  // derived-class pointer

   // set floating-point numeric formatting
   cout << fixed << setprecision( 2 );

   // output objects point and circle
   cout << "Print point and circle objects:"
        << "\nPoint: ";
   point.print();   // invokes Point's print
   cout << "\nCircle: ";
   circle.print();  // invokes Circle's print 

   // aim base-class pointer at base-class object and print
   pointPtr = &point;
   cout << "\n\nCalling print with base-class pointer to " 
        << "\nbase-class object invokes base-class print "
        << "function:\n";
   pointPtr->print();  // invokes Point's print

   // aim derived-class pointer at derived-class object
   // and print
   circlePtr = &circle;
   cout << "\n\nCalling print with derived-class pointer to "
        << "\nderived-class object invokes derived-class "
        << "print function:\n";
   circlePtr->print();  // invokes Circle's print

   // aim base-class pointer at derived-class object and print
   pointPtr = &circle;
   cout << "\n\nCalling print with base-class pointer to " 
        << "derived-class object\ninvokes base-class print "
        << "function on that derived-class object\n";
   pointPtr->print();  // invokes Point's print
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