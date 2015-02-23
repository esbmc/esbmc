// Fig. 7.41: elevator.h
// Elevator class definition.
#ifndef ELEVATOR_H
#define ELEVATOR_H

#include "elevatorButton.h"
#include "door.h"
#include "bell.h"

class Floor;   // forward declaration
class Person;  // forward declaration

class Elevator {

public:
   Elevator( Floor &, Floor & );   // constructor
   ~Elevator();                    // destructor
   void summonElevator( int );     // request to service floor
   void prepareToLeave( bool );    // prepare to leave
   void processTime( int );    // give current time to elevator
   void passengerEnters( Person * const ); // board a passenger
   void passengerExits();          // exit a passenger

   // public object accessible to client code with 
   // access to Elevator object
   ElevatorButton elevatorButton;  

private:

   // utility functions
   void processPossibleArrival();
   void processPossibleDeparture();
   void arriveAtFloor( Floor & );
   void move();

   // static constants that represent time required to travel 
   // between floors and directions of the elevator
   static const int ELEVATOR_TRAVEL_TIME;  
   static const int UP;                    
   static const int DOWN;                  

   // data members
   int currentBuildingClockTime;  // current time
   bool moving;                   // elevator state
   int direction;                 // current direction
   int currentFloor;              // current location
   int arrivalTime;               // time to arrive at a floor
   bool floor1NeedsService;       // floor1 service flag
   bool floor2NeedsService;       // floor2 service flag

   Floor &floor1Ref;              // reference to floor1
   Floor &floor2Ref;              // reference to floor2
   Person *passengerPtr;          // pointer to passenger
   
   Door door;                     // door object
   Bell bell;                     // bell object

}; // end class Elevator

#endif // ELEVATOR_H

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
