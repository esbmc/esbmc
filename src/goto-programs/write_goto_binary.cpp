/*******************************************************************\

Module: Write GOTO binaries

Author: CM Wintersteiger

\*******************************************************************/

#include <fstream>
#include <goto-programs/goto_function_serialization.h>
#include <goto-programs/write_goto_binary.h>
#include <util/irep_serialization.h>
#include <util/message.h>
#include <util/symbol_serialization.h>

bool write_goto_binary(
  std::ostream &out,
  const contextt &lcontext,
  goto_functionst &functions)
{
  // header
  out << "GBF";
  write_long(out, GOTO_BINARY_VERSION);

  irep_serializationt::ireps_containert irepc;
  irep_serializationt irepconverter(irepc);
  symbol_serializationt symbolconverter(irepc);
  goto_function_serializationt gfconverter(irepc);

  write_long(out, lcontext.size());

  lcontext.foreach_operand([&symbolconverter, &out](const symbolt &s) {
    symbolconverter.convert(s, out);
  });

  unsigned cnt = 0;
  forall_goto_functions(it, functions)
    if(it->second.body_available)
      cnt++;

  write_long(out, cnt);

  for(auto &it : functions.function_map)
  {
    if(it.second.body_available)
    {
      it.second.body.compute_location_numbers();
      write_string(out, it.first.as_string());
      gfconverter.convert(it.second, out);
    }
  }

  //irepconverter.output_map(f);
  //irepconverter.output_string_map(f);

  return false;
}
