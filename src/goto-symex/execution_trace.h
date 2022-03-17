#ifndef EXECUTION_TRACE_H
#define EXECUTION_TRACE_H

#include <vector>
#include <string>
#include <goto-programs/goto_program.h>
#include <util/namespace.h>


class c_instructiont : public goto_programt::instructiont
{

public:
  std::string msg;

  c_instructiont() : goto_programt::instructiont()
  {
  }

  c_instructiont(goto_programt::instructiont &i) : goto_programt::instructiont(i)
  {
  }
  
  c_instructiont(const goto_programt::instructiont &i) : goto_programt::instructiont(i)
  {
  }

  virtual ~c_instructiont() = default;

  std::string convert_to_c(namespacet &ns);

protected:
  
  std::string convert_assert_to_c(namespacet &ns);
  std::string convert_other_to_c(namespacet &ns);

};

extern std::vector<c_instructiont> instructions_to_c;

#endif
