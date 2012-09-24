// Fig. 7.29: scheduler.h
// Scheduler class definition.
#ifndef SCHEDULER_H
#define SCHEDULER_H

class Floor;                        // forward declaration

class Scheduler {

public:
   Scheduler( Floor &, Floor & );   // constructor 
   ~Scheduler();                    // destructor
   void processTime( int );         // set scheduler's time

private:
   // schedule arrival to a floor
   void scheduleTime( const Floor & );

   // delay arrival to a floor
   void delayTime( const Floor & );

   // create new person; place on floor
   void createNewPerson( Floor & );   
   
   // handle person arrival on a floor
   void handleArrivals( Floor &, int );   

   int currentClockTime;

   Floor &floor1Ref;
   Floor &floor2Ref;

   int floor1ArrivalTime;
   int floor2ArrivalTime;

}; // end class Scheduler

#endif // SCHEDULER_H

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
