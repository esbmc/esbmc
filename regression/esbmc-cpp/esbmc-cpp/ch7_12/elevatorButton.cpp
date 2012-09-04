// Fig. 7.38: elevatorButton.cpp: 
// Member-function definitions for class ElevatorButton.
#include <iostream>

using std::cout;
using std::endl;

#include "elevatorButton.h"  // ElevatorButton class definition
#include "elevator.h"        // Elevator class definition

// constructor
ElevatorButton::ElevatorButton( Elevator &elevatorHandle )
   : pressed( false ), 
     elevatorRef( elevatorHandle )
{ 
   cout << "elevator button constructed" << endl; 

} // end ElevatorButton constructor

// destructor
ElevatorButton::~ElevatorButton() 
{ 
   cout << "elevator button destructed" << endl; 

} // end ~ElevatorButton destructor

// press button and signal elevator to prepare to leave floor
void ElevatorButton::pressButton() 
{
   pressed = true;
   cout << "elevator button tells elevator to prepare to leave" 
        << endl;
   elevatorRef.prepareToLeave( true );

} // end function pressButton

// reset button
void ElevatorButton::resetButton() 
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
