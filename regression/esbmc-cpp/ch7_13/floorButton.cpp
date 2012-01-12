// Fig. 7.40: floorButton.cpp
// Member-function definitions for class FloorButton.
#include <iostream>

using std::cout;
using std::endl;

#include "floorButton.h"
#include "elevator.h"

// constructor
FloorButton::FloorButton( int floor, Elevator &elevatorHandle )
   : floorNumber( floor ), 
     pressed( false ), 
     elevatorRef( elevatorHandle ) 
{ 
   cout << "floor " << floorNumber << " button constructed"
        << endl; 

} // end FloorButton constructor

// destructor
FloorButton::~FloorButton() 
{
   cout << "floor " << floorNumber << " button destructed"
        << endl;

} // end ~FloorButton destructor

// press the button
void FloorButton::pressButton()
{
   pressed = true;
   cout << "floor " << floorNumber 
        << " button summons elevator" << endl;

   // call elevator to this floor
   elevatorRef.summonElevator( floorNumber ); 

} // end function pressButton

// reset button
void FloorButton::resetButton() 
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
