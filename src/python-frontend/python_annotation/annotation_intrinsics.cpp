#include <python-frontend/python_annotation/annotation_intrinsics.h>

namespace python_annotation_intrinsics
{
const std::map<std::string, std::string> &builtin_functions()
{
  // The entries below are a verbatim transcription of the original
  // namespace-scope `builtin_functions` constant from python_annotation.h.
  // Keep them byte-identical and in the same order.
  static const std::map<std::string, std::string> table = {
    // Type conversion functions
    {"int", "int"},
    {"float", "float"},
    {"str", "str"},
    {"bool", "bool"},
    {"list", "list"},
    {"dict", "dict"},
    {"set", "set"},
    {"tuple", "tuple"},

    // Numeric functions
    {"abs", "int"},   // Can return int or float, but int is common case
    {"round", "int"}, // Can return int or float
    {"min", "Any"},   // Type depends on input
    {"max", "Any"},   // Type depends on input
    {"sum", "Any"},   // Type depends on input
    {"pow", "int"},   // Can return int or float

    // Sequence functions
    {"len", "int"},
    {"range", "range"},
    {"enumerate", "enumerate"},
    {"zip", "zip"},
    {"reversed", "list"},
    {"sorted", "list"},

    // I/O functions
    {"print", "NoneType"},
    {"input", "str"},
    {"open", "file"},

    // Utility functions
    {"isinstance", "bool"},
    {"issubclass", "bool"},
    {"hasattr", "bool"},
    {"getattr", "Any"},
    {"setattr", "NoneType"},
    {"delattr", "NoneType"},
    {"callable", "bool"},
    {"id", "int"},
    {"hash", "int"},
    {"repr", "str"},
    {"ascii", "str"},
    {"ord", "int"},
    {"chr", "str"},
    {"bin", "str"},
    {"oct", "str"},
    {"hex", "str"},
    {"format", "str"},

    // Iteration functions
    {"iter", "iterator"},
    {"next", "Any"},
    {"all", "bool"},
    {"any", "bool"},
    {"filter", "filter"},
    {"map", "map"},

    // Variable functions
    {"vars", "dict"},
    {"dir", "list"},
    {"globals", "dict"},
    {"locals", "dict"},

    // Python threading.Thread lowering (see parser.py
    // lower_threading_thread_usage and src/c2goto/library/pthread_lib.c).
    // Annotated here so the call's return type is known during inference
    // even though the C definitions are linked in by c2goto, not declared
    // in any Python model.
    {"__ESBMC_spawn_thread", "int"},
    {"__pyt_init_tid", "NoneType"},
    {"__pyt_join", "NoneType"},
    {"__pyt_terminate", "NoneType"},
    {"__ESBMC_pylock_block_and_check", "NoneType"},

    // Execution functions
    {"eval", "Any"},
    {"exec", "NoneType"},
    {"compile", "code"},

    // String module constants
    {"string.digits", "str"},
    {"string.ascii_lowercase", "str"},
    {"string.ascii_uppercase", "str"},
    {"string.ascii_letters", "str"},
    {"string.punctuation", "str"},
    {"string.whitespace", "str"},
    {"string.printable", "str"},
    {"string.hexdigits", "str"},
    {"string.octdigits", "str"},

    // Import functions
    {"__import__", "module"}};
  return table;
}

} // namespace python_annotation_intrinsics
