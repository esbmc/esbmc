#pragma once

#include <nlohmann/json.hpp>

#include <string>

// Pure, state-free helpers extracted from python_annotation.h. Each
// function below is a verbatim move of a previously class-local or
// namespace-scope helper; no semantic change is intended. The helpers
// take `const nlohmann::json &` directly — the original code used a
// `Json` template parameter that was only ever instantiated with
// `nlohmann::json`.
namespace python_annotation_utils
{
// Outcome of type inference. Used by callers of the type-inference
// dispatcher to distinguish a successfully resolved type from an
// unknown one.
enum class InferResult
{
  OK,
  UNKNOWN,
};

// Concrete builtin Python type names produced by inspecting a JSON
// `Constant.value` node. Returns "null", "bool", "int", "float",
// "str", "array", "object" or "unknown" — the exact strings the
// previous implementation produced. Callers depend on these spellings.
std::string get_type_from_json(const nlohmann::json &value);

// Returns true when `node` carries a non-null `annotation` field.
// `node.empty()` is also accepted as input and yields false.
bool has_annotation(const nlohmann::json &node);

// True when `body` contains any explicit `return` / `return None` —
// either at the top level or inside nested `body`/`orelse` blocks.
bool has_return_none(const nlohmann::json &body);

// Recursively reconstructs a dotted base name for an attribute /
// subscript chain rooted at a Name node (e.g. `obj.attr[0]`).
// Returns the empty string on unrecognised shapes.
std::string get_base_var_name(const nlohmann::json &node);

// Splits `input` on '.', reverses the resulting tokens and rejoins
// them with '.'. Pure string utility; behaviour must match the
// original `invert_substrings` byte-for-byte for the values the
// annotation pass feeds it.
std::string invert_substrings(const std::string &input);

// Inspects the *second* element of a function-default-arg list and
// returns a guess of the runtime container/scalar type ("list",
// "dict", "set", "tuple", "int", "float", "str", "bool", "None")
// or "Any" if the shape is unrecognised. Returns an empty string
// when `args_node` is not an array of at least two entries.
std::string infer_type_from_default_arg_shape(const nlohmann::json &args_node);

} // namespace python_annotation_utils
