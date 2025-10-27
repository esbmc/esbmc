// Fig. 7.35: door.h
// Door class definition.
#ifndef DOOR_H
#define DOOR_H

class Person;    // forward declaration
class Floor;     // forward declaration
class Elevator;  // forward declaration

class Door {

public:
   Door();       // constructor
   ~Door();      // destructor
  
   void openDoor( Person * const,             // opens door
      Person * const, Floor &, Elevator & ); 
   void closeDoor( const Floor & );           // closes door
   
private:
   bool open;    // open or closed

};

#endif // DOOR_H

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
