// Fig. 16.21: login.cpp
// Program to output an XHTML form, verify the
// username and password entered, and add members.
#include <iostream>

using std::cerr;
using std::cout;
using std::cin;
using std::ios;

#include <fstream>

using std::ifstream;
using std::ofstream;

#include <string>

using std::string;

#include <cstdlib>

void header();
void writeCookie();

int main()
{
   char query[ 1024 ] = "";
   string dataString = "";
   
   // strings to store username and password
   string userName = "";
   string passWord = "";
   string newCheck = "";
   
   int contentLength = 0;
   int endPassword = 0;

   // data was posted
   if ( getenv( "CONTENT_LENGTH" ) ) {
      // retrieve query string
      contentLength = atoi( getenv( "CONTENT_LENGTH" ) );
      cin.read( query, contentLength );
      dataString = query;

      
      // find username location
      int userLocation = dataString.find( "user=" ) + 5;
      int endUser = dataString.find( "&" );
      
      // find password location
      int passwordLocation = dataString.find( "password=" ) + 9;

      endPassword = dataString.find( "&new" );
      
      // new membership requested
      if ( endPassword > 0 )
         passWord = dataString.substr( passwordLocation, 
            endPassword - passwordLocation );
      
      // existing member 
      else
         passWord = dataString.substr( passwordLocation );

      userName = dataString.substr( userLocation, endUser - 
         userLocation );
   } // end if
   // no data was retrieved
   if ( dataString == "" ) {
      header();
      cout << "<p>Please login.</p>";
      
      // output login form
      cout << "<form method = \"post\" " 
           << "action = \"/cgi-bin/login.cgi\"><p>"
           << "User Name: " 
           << "<input type = \"text\" name = \"user\"/><br/>"
           << "Password: " 
           << "<input type = \"password\" " 
           << "name = \"password\"/><br/>"
           << "New? <input type = \"checkbox\""
           << " name = \"new\" " 
           << "value = \"1\"/></p>"
           << "<input type = \"submit\" value = \"login\"/>" 
           << "</form>";

   } // end if
   
   // process entered data
   else {

      // add new member 
      if ( endPassword > 0 ) {
         string fileUsername = "";
         string filePassword = "";
         bool nameTaken = false;
         
         // open password file
         ifstream userData( "userdata.txt", ios::in );
         
         // could not open file
         if ( !userData ) {
            cerr << "Could not open database.";
            exit( 1 );
         } // end if
         
         // read username and password from file
         while ( userData >> fileUsername >> filePassword ) {

            // name is already taken
            if ( userName == fileUsername )
               nameTaken = true;

         } // end while
         
         // user name is taken
         if ( nameTaken ) {
            header();
            cout << "<p>This name has already been taken.</p>"
                 << "<a href=\"/cgi-bin/login.cgi\">" 
                 << "Try Again</a>";
         } // end if
         
         // process data
         else {
            
            // write cookie
            writeCookie();
            header();
            
            // open user data file
            ofstream userData( "userdata.txt", ios::app );
            
            // could not open file
            if ( !userData ) {
               cerr << "Could not open database.";
               exit( 1 );
            } // end if
            
            // write user data to file
            userData << "\n" << userName << "\n" << passWord;
            
            cout << "<p>Your information has been processed."
                 << "<a href=\"/cgi-bin/shop.cgi\">" 
                 << "Start Shopping</a></p>";

         } // end else
      } // end if

      // search for password if entered
      else {

         // strings to store username and password from file
         string fileUsername = "";
         string filePassword = "";
         bool authenticated = false;
         bool userFound = false;
         
         // open password file
         ifstream userData( "userdata.txt", ios::in );
         
         // could not open file
         if ( !userData ) {
            cerr << "Could not open database.";
            exit( 1 );
         } // end if
         
         // read in user data
         while ( userData >> fileUsername >> filePassword ) {

            // username and password match
            if ( userName == fileUsername && 
               passWord == filePassword )
               authenticated = true;
            
            // username was found
            if ( userName == fileUsername )
               userFound = true;
         } // end while 
         
         // user is authenticated
         if ( authenticated ) {
            writeCookie();
            header();

            cout << "<p>Thank you for returning, " 
                 << userName << "!</p>"
                 << "<a href=\"/cgi-bin/shop.cgi\">" 
                 << "Start Shopping</a>";
         } // end if

         // user not authenticated
         else {
            header();
            
            // password is incorrect
            if ( userFound )
               cout << "<p>You have entered an incorrect " 
                    << "password. Please try again.</p>"
                    << "<a href=\"/cgi-bin/login.cgi\">" 
                    << "Back to login</a>";            

            // user is not registered
            else 
               cout << "<p>You are not a registered user.</p>"
                    << "<a href=\"/cgi-bin/login.cgi\">" 
                    << "Register</a>";

         } // end else 
      } // end else 
   } // end if

   cout << "</body>\n</html>\n";
   return 0;
} // end main

// function to output header
void header()
{
   // output header
   cout << "Content-type: text/html\n\n";
   
   // output XML declaration and DOCTYPE
   cout << "<?xml version = \"1.0\"?>"
        << "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 "
        << "Transitional//EN\" \"http://www.w3.org/TR/xhtml1"
        << "/DTD/xhtml1-transitional.dtd\">";

   // output html element and some of its contents
   cout << "<html xmlns = \"http://www.w3.org/1999/xhtml\">"
        << "<head><title>Login Page</title></head>"
        << "<body>";

} // end header

// function to write cookie data
void writeCookie()
{
   string expires = "Friday, 14-MAY-04 16:00:00 GMT";
   cout << "set-cookie: CART=; expires=" 
        << expires << "; path=\n";

} // end writeCookie

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
