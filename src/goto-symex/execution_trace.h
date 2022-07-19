#ifndef EXECUTION_TRACE_H
#define EXECUTION_TRACE_H

#include <vector>
#include <string>
#include <goto-programs/goto_program.h>
#include <goto-programs/goto_functions.h>
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

  unsigned int get_loop_depth();
  void set_loop_depth(unsigned int depth);
  void increment_loop_depth();

  codet decl;

  bool scope_begin = false;
  bool scope_end = false;

protected:

  unsigned int loop_depth = 0;
  unsigned int fun_call_num = 0;

  std::string convert_assert_to_c(namespacet &ns);
  std::string convert_decl_to_c(namespacet &ns);
  std::string convert_other_to_c(namespacet &ns);
};

void inline_function_calls(std::vector<c_instructiont> c_instructions, goto_functionst goto_functions);
void insert_static_declarations(std::vector<c_instructiont> c_instructions, namespacet ns);
void merge_decl_assign_pairs(std::vector<c_instructiont> c_instructions);
void assign_returns(std::vector<c_instructiont> c_instructions);

void assign_dynamic_sizes(std::vector<c_instructiont> c_instructions);

void adjust_gotos(std::vector<c_instructiont> c_instructions, namespacet ns);

std::vector<c_instructiont> inline_function_call(c_instructiont func_call, goto_functionst goto_functions);

extern std::vector<c_instructiont> instructions_to_c;
extern std::map<std::string, std::vector<c_instructiont>> functions_to_c;
extern unsigned int function_call_num;
extern unsigned int label_num;
extern std::map<std::string, unsigned int> fun_call_nums;

extern std::map<std::string, type2tc> alive_vars;
extern std::vector<std::string> declared_types;


#endif
