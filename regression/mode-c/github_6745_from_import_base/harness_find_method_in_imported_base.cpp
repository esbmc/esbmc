// Mode C (C-Live) reachability harness for esbmc/esbmc commit 5bcd7a9de0
// (issue #6745, "[python] Keep a local class ahead of a same-named import").
//
// Production site: src/python-frontend/converter/converter_symbols.cpp,
// python_converter::find_method_in_imported_base(), added branch
// (post-patch lines 152-158):
//
//   else if (
//     json_utils::find_class((*ast_json)["body"], base_class) ==
//     nlohmann::json())
//   {
//     const auto binding = from_import_binding(*ast_json, base_class);
//     module_path = binding.first;
//     defined_name = binding.second;
//   }
//
// FULL-FIDELITY HARNESS DECLARED INFEASIBLE: find_method_in_imported_base is
// a `const` member of python_converter (src/python-frontend/python_converter.h)
// reading `ast_json` (a parsed nlohmann::json AST) and `symbol_table_` (a full
// ESBMC symbol table) -- not a small dependency surface. nlohmann::json
// itself is independently infeasible under ESBMC's own C++ frontend: probed
// directly with
//   esbmc probe_json.cpp -I <build>/_deps/json-src/single_include \
//     --std c++17 --goto-functions-only
// ESBMC's bundled <map>/<memory> operational models reject the
// allocator_traits::construct / std::map::emplace patterns basic_json's
// array/object constructors rely on ("no matching member function for call
// to 'construct'" / "'insert'") -> PARSING ERROR. This matches the task's own
// steer that a whole-program proof is not realistic here.
//
// This harness lifts the guard to the two booleans it evaluates, each
// backed by a citation for why it is independently realizable:
//
//   qualified -- converter_symbols.cpp:140-142. Determined solely by the
//     shape of base_class_node, one entry of class_node["bases"]. Python's
//     ast.Name node (bare `Base`) has no `value` field
//     (https://docs.python.org/3/library/ast.html#ast.Name), so
//     contains("value") is false and qualified is false regardless of any
//     other AST content. ast.Attribute (`mod.Base`) carries `value` as a
//     Name/Attribute object, making qualified true. qualified therefore
//     cannot depend on whether base_class is also defined locally.
//
//   base_not_defined_locally -- converter_symbols.cpp:152-153,
//     `find_class((*ast_json)["body"], base_class) == nlohmann::json()`.
//     find_class (json_utils.h:60-70) is a pure, patch-unrelated function of
//     the file's own top-level statement list; it is true iff no ClassDef
//     entry in that list is named base_class -- a different AST subtree
//     from base_class_node, hence independent of `qualified`.
//
// Realizability of qualified==false && base_not_defined_locally==true is
// pinned end-to-end by regression/python/github_6745_from_import_ctor
// (`from shapes import Shape`, no local `class Shape` in main.py); the
// guard's false arm (qualified==false, base_not_defined_locally==false) is
// pinned by regression/python/github_6745_from_import_shadowed_fail (same
// import, but `class Shape` also defined in main.py).

extern "C" bool nondet_bool();

struct binding_t
{
  const char *module_path;
  const char *defined_name;
};

// G4 stand-in for from_import_binding (converter_symbols.cpp:104-128): a
// pure function of the AST and the bound name, orthogonal to reachability of
// the call site itself. Pure-nondet, no masking/narrowing of its result.
static binding_t from_import_binding_stub()
{
  binding_t b;
  b.module_path = nondet_bool() ? "shapes.py" : "";
  b.defined_name = "Shape";
  return b;
}

static const char *get_imported_module_path_stub()
{
  return "unused";
}

int main()
{
  bool qualified = nondet_bool();
  bool base_not_defined_locally = nondet_bool();

  // G3 preconditions -- see file header citations.
  __ESBMC_assume(!qualified);
  __ESBMC_assume(base_not_defined_locally);

  const char *module_path = "";
  const char *defined_name = "base_class_placeholder";

  if (qualified)
  {
    module_path = get_imported_module_path_stub();
  }
  else if (base_not_defined_locally)
  {
    __ESBMC_unreachable();
    binding_t binding = from_import_binding_stub();
    module_path = binding.module_path;
    defined_name = binding.defined_name;
  }

  (void)module_path;
  (void)defined_name;
  return 0;
}
