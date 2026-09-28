#pragma once

#include <nlohmann/json.hpp>
#include <vector>

namespace python_param_annotations
{
/// One module's AST. @c write is null for modules that may be read (to find
/// call sites) but not rewritten. @c name is the module's import name, empty
/// for the entry-point module.
struct module_ast
{
  const nlohmann::json *read;
  nlohmann::json *write;
  std::string name;
};

/**
 * @brief Retype `list[...]` parameters from the tuples their callers pass.
 *
 * Python does not enforce annotations, so `def f(a: list[int])` fed
 * `[(1, 5), (1, 2)]` would have each tuple's payload reinterpreted as an int
 * and report VERIFICATION SUCCESSFUL for a program CPython rejects
 * (GitHub #5936). Rewriting the parameter's annotation to the element type its
 * callers actually pass corrects every consumer at once, and must therefore run
 * before the AST is type-annotated.
 *
 * A parameter fed different element types by different call sites, or one whose
 * argument this pass cannot type, is left alone: the frontend emits one GOTO
 * function per Python function, so no single annotation could be correct.
 * Only tuple element types are inferred; a mis-annotated scalar is already
 * corrected at read time by the stored-type_id dispatch in the list model.
 *
 * @param modules Every module in the import graph. Call sites are searched in
 *                all of them; only those with a non-null @c write are rewritten.
 */
void propagate_tuple_list_params(const std::vector<module_ast> &modules);
} // namespace python_param_annotations
