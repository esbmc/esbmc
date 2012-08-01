// Fig. 10.19: cylinder.cpp
// Cylinder class inherits from class Circle.
#include <iostream>

using std::cout;

#include "cylinder.h"   // Cylinder class definition

// default constructor
Cylinder::Cylinder( int xValue, int yValue, double radiusValue, 
  double heightValue ) 
  : Circle( xValue, yValue, radiusValue )
{
   setHeight( heightValue );

} // end Cylinder constructor

// set Cylinder's height
void Cylinder::setHeight( double heightValue )
{
   height = ( heightValue < 0.0 ? 0.0 : heightValue );

} // end function setHeight

// get Cylinder's height
double Cylinder::getHeight() const
{
   return height;

} // end function getHeight

// override virtual function getArea: return Cylinder area
double Cylinder::getArea() const
{
   return 2 * Circle::getArea() +           // code reuse
      getCircumference() * getHeight();

} // end function getArea

// override virtual function getVolume: return Cylinder volume
double Cylinder::getVolume() const
{
   return Circle::getArea() * getHeight();  // code reuse

} // end function getVolume

// override virtual function getName: return name of Cylinder
string Cylinder::getName() const
{
   return "Cylinder";

}  // end function getName

// override virtual function print: output Cylinder object
void Cylinder::print() const
{
   Circle::print();  // code reuse
   cout << "; height is " << getHeight();

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