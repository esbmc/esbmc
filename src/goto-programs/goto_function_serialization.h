#ifndef GOTO_FUNCTION_SERIALIZATION_H_
#define GOTO_FUNCTION_SERIALIZATION_H_

#include <goto-programs/goto_functions.h>
#include <goto-programs/goto_program_serialization.h>
#include <util/irep_serialization.h>

class goto_function_serializationt
{
private:
  goto_program_serializationt gpconverter;

public:
  goto_function_serializationt(irep_serializationt::ireps_containert &ic)
    : gpconverter(ic){};

  void convert(std::istream &, irept &);
  void convert(const goto_functiont &, std::ostream &);
};

#endif /*GOTO_FUNCTION_SERIALIZATION_H_*/
