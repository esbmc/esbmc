// Fig. 9.34: button.cpp
// Member function definitions for class Button.
#include <iostream>

using std::cout;
using std::endl;

#include "button.h"  // Button class definition

// constructor
Button::Button( Elevator &elevatorHandle )
   : elevatorRef( elevatorHandle ), pressed( false )
{ 
   cout << "button constructed" << endl; 

} // end Button constructor

// destructor
Button::~Button() 
{ 
   cout << "button destructed" << endl; 

} // end Button destructor

// press button
void Button::pressButton() 
{ 
   pressed = true; 

} // end function pressButton

// reset button
void Button::resetButton() 
{ 
   pressed = false; 

} // end function resetButton

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
