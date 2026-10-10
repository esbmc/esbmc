#pragma once

#include <goto-programs/goto_functions.h>
#include <llm/goto/scope.h>

#include <map>
#include <string>

namespace llm
{
/// A GOTO function as read-only pseudo-C for a language model.
struct rendered_functiont
{
  std::string text;
  scopet scope;
  std::map<std::string, goto_programt::const_targett> loops;
};

rendered_functiont render_function(
  const irep_idt &name,
  const goto_functiont &function,
  const namespacet &ns);

std::string
render_expr(const expr2tc &expr, const scopet &scope, const namespacet &ns);
} // namespace llm
