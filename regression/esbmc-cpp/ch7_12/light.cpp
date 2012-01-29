// Fig. 7.34: light.cpp
// Member-function definitions for class Light.
#include <iostream>

using std::cout;
using std::endl;

#include "light.h"  // Light class definition

// constructor
Light::Light( int number ) 
   : on( false ), 
     floorNumber( number )
{ 
   cout << "floor " << floorNumber << " light constructed" 
        << endl;

} // end Light constructor

// destuctor
Light::~Light() 
{ 
   cout << "floor " << floorNumber 
        << " light destructed" << endl;

} // end ~Light destructor

// turn light on
void Light::turnOn() 
{ 
   if ( !on ) {  // if light not on, turn it on
      on = true; 
      cout << "floor " << floorNumber 
           << " light turns on" << endl;

   } // end if

} // end function turnOn

// turn light off
void Light::turnOff() 
{ 
   if ( on ) {  // if light is on, turn it off
      on = false; 
      cout << "floor " << floorNumber 
           << " light turns off" << endl;

   } // end if

} // end function turnOff

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
