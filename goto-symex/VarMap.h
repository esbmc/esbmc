//Bruno J. Savino

#ifndef VARMAP_H
#define VARMAP_H

#include <iostream>

class VarMap {

public:

   // default VarMap constructor
   VarMap( std::string = "", std::string = "" );

   // accessor functions for VarName
   void setVarName( std::string );
   std::string getVarName() const;

   // accessor functions for VarInfo
   void setVarInfo( std::string );
   std::string getVarInfo() const;

private:
   char varName[50];
   char varInfo[30];


}; // end class VarMap

#endif
