// Fig. 3.9: GradeBook.h
// GradeBook class definition in a separate file from main.
#include <iostream> 
#include <string> // class GradeBook uses C++ standard string class

// GradeBook class definition
class GradeBook
{
public:
   // constructor initializes courseName with string supplied as argument
   explicit GradeBook( std::string name )
      : courseName( name ) // member initializer to initialize courseName 
   {
      // empty body
   } // end GradeBook constructor

   // function to set the course name
   void setCourseName( std::string name )
   {
      courseName = name; // store the course name in the object
   } // end function setCourseName

   // function to get the course name
   std::string getCourseName() const
   {
      return courseName; // return object's courseName
   } // end function getCourseName

   // display a welcome message to the GradeBook user
   void displayMessage() const
   {
      // call getCourseName to get the courseName
      std::cout << "Welcome to the grade book for\n" << getCourseName()  
         << "!" << std::endl;
   } // end function displayMessage
private:
   std::string courseName; // course name for this GradeBook
}; // end class GradeBook  

/**************************************************************************
 * (C) Copyright 1992-2012 by Deitel & Associates, Inc. and               *
 * Pearson Education, Inc. All Rights Reserved.                           *
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
 **************************************************************************/   
