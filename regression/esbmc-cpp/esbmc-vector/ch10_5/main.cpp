// Fig. 10.20: fig10_20.cpp
// Driver for shape, point, circle, cylinder hierarchy.
#include <iostream>

using std::cout;
using std::endl;
using std::fixed;

#include <iomanip>

using std::setprecision;

#include <vector>

using std::vector;

#include "shape.h"     // Shape class definition
#include "point.h"     // Point class definition 
#include "circle.h"    // Circle class definition 
#include "cylinder.h"  // Cylinder class definition

void virtualViaPointer( const Shape * );  
void virtualViaReference( const Shape & );

int main()
{
   // set floating-point number format
   cout << fixed << setprecision( 2 );

   Point point( 7, 11 );                  // create a Point
   Circle circle( 22, 8, 3.5 );           // create a Circle
   Cylinder cylinder( 10, 10, 3.3, 10 );  // create a Cylinder

   cout << point.getName() << ": ";    // static binding
   point.print();                      // static binding
   cout << '\n';

   cout << circle.getName() << ": ";   // static binding
   circle.print();                     // static binding
   cout << '\n';

   cout << cylinder.getName() << ": "; // static binding
   cylinder.print();                   // static binding
   cout << "\n\n";

   // create vector of three base-class pointers
   vector< Shape * > shapeVector( 3 );  

   // aim shapeVector[0] at derived-class Point object
   shapeVector[ 0 ] = &point;

   // aim shapeVector[1] at derived-class Circle object
   shapeVector[ 1 ] = &circle;

   // aim shapeVector[2] at derived-class Cylinder object
   shapeVector[ 2 ] = &cylinder;

   // loop through shapeVector and call virtualViaPointer
   // to print the shape name, attributes, area and volume 
   // of each object using dynamic binding
   cout << "\nVirtual function calls made off "
        << "base-class pointers:\n\n";

   for ( int i = 0; i < shapeVector.size(); i++ ) 
      virtualViaPointer( shapeVector[ i ] );

   // loop through shapeVector and call virtualViaReference
   // to print the shape name, attributes, area and volume 
   // of each object using dynamic binding
   cout << "\nVirtual function calls made off "
        << "base-class references:\n\n";

   for ( int j = 0; j < shapeVector.size(); j++ ) 
      virtualViaReference( *shapeVector[ j ] );

   return 0;

} // end main

// make virtual function calls off a base-class pointer
// using dynamic binding
void virtualViaPointer( const Shape *baseClassPtr )
{
   cout << baseClassPtr->getName() << ": ";

   baseClassPtr->print();

   cout << "\narea is " << baseClassPtr->getArea()
        << "\nvolume is " << baseClassPtr->getVolume() 
        << "\n\n";

} // end function virtualViaPointer

// make virtual function calls off a base-class reference
// using dynamic binding
void virtualViaReference( const Shape &baseClassRef )
{
   cout << baseClassRef.getName() << ": ";

   baseClassRef.print();

   cout << "\narea is " << baseClassRef.getArea()
        << "\nvolume is " << baseClassRef.getVolume() << "\n\n";

} // end function virtualViaReference

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