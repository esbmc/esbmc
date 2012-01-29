// Fig. 7.36: door.cpp
// Member-function definitions for class Door.
#include <iostream>

using std::cout;
using std::endl;

#include "door.h"      // Door class definition
#include "person.h"    // Person class definition
#include "floor.h"     // Floor class definition
#include "elevator.h"  // Elevator class definition

// constructor
Door::Door() 
   : open( false )  // initialize open to false
{ 
   cout << "door constructed" << endl; 

} // end Door constructor

// destructor
Door::~Door() 
{
   cout << "door destructed" << endl; 

} // end ~Door destructor

// open the door
void Door::openDoor( Person * const passengerPtr, 
   Person * const nextPassengerPtr, Floor &currentFloor, 
   Elevator &elevator ) 
{ 
   if ( !open ) {  // if door is not open, open door
      open = true; 
   
      cout << "elevator opens its door on floor "
           << currentFloor.getNumber() << endl;

      // if passenger is in elevator, tell person to leave
      if ( passengerPtr != 0 ) {
         passengerPtr->exitElevator( currentFloor, elevator );
         delete passengerPtr; // passenger leaves simulation

      } // end if

      // if passenger waiting to enter elevator, 
      // tell passenger to enter
      if ( nextPassengerPtr != 0 )
         nextPassengerPtr->enterElevator( 
            elevator, currentFloor );

   } // end outer if

} // end function openDoor

// close the door
void Door::closeDoor( const Floor &currentFloor ) 
{
   if ( open ) {  // if door is open, close door
      open = false; 
      cout << "elevator closes its door on floor "
           << currentFloor.getNumber() << endl;

   } // end if

} // end function closeDoor

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
