// Fig. 9.24: cylindertest.cpp
// Testing class Cylinder.
#include <iostream>  

using std::cout;
using std::endl;
using std::fixed;

#include <iomanip>

using std::setprecision;

#include "cylinder.h"  // Cylinder class definition

int main()
{
   // instantiate Cylinder object
   Cylinder cylinder( 12, 23, 2.5, 5.7 ); 

   // display point coordinates
   cout << "X coordinate is " << cylinder.getX() 
        << "\nY coordinate is " << cylinder.getY()
        << "\nRadius is " << cylinder.getRadius()
        << "\nHeight is " << cylinder.getHeight();

   cylinder.setX( 2 );          // set new x-coordinate
   cylinder.setY( 2 );          // set new y-coordinate
   cylinder.setRadius( 4.25 );  // set new radius
   cylinder.setHeight( 10 );    // set new height

   // display new cylinder value
   cout << "\n\nThe new location and radius of circle are\n";
   cylinder.print();

   // display floating-point values with 2 digits of precision
   cout << fixed << setprecision( 2 );

   // display cylinder's diameter
   cout << "\n\nDiameter is " << cylinder.getDiameter();

   // display cylinder's circumference
   cout << "\nCircumference is " 
        << cylinder.getCircumference();

   // display cylinder's area
   cout << "\nArea is " << cylinder.getArea();

   // display cylinder's volume
   cout << "\nVolume is " << cylinder.getVolume();

   cout << endl;

   return 0;  // indicates successful termination
   
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
