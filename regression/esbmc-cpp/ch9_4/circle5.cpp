// Fig. 9.28: circle5.cpp
// Circle5 class member-function definitions.
#include <iostream>  

using std::cout;
using std::endl;

#include "circle5.h"   // Circle5 class definition

// default constructor
Circle5::Circle5( int xValue, int yValue, double radiusValue )
   : Point4( xValue, yValue )  // call base-class constructor
{
   setRadius( radiusValue );

   cout << "Circle5 constructor: ";
   print();
   cout << endl;

} // end Circle5 constructor

// destructor
Circle5::~Circle5()
{
   cout << "Circle5 destructor: ";
   print();
   cout << endl;

} // end Circle5 destructor

// set radius 
void Circle5::setRadius( double radiusValue )
{
   radius = ( radiusValue < 0.0 ? 0.0 : radiusValue );

} // end function setRadius

// return radius 
double Circle5::getRadius() const
{
   return radius;

} // end function getRadius

// calculate and return diameter
double Circle5::getDiameter() const
{
   return 2 * getRadius();

} // end function getDiameter

// calculate and return circumference
double Circle5::getCircumference() const
{
   return 3.14159 * getDiameter();

} // end function getCircumference

// calculate and return area
double Circle5::getArea() const
{
   return 3.14159 * getRadius() * getRadius();

} // end function getArea

// output Circle5 object
void Circle5::print() const
{
   cout << "Center = ";
   Point4::print();      // invoke Point4's print function
   cout << "; Radius = " << getRadius();

} // end function print

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