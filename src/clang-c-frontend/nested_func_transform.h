#pragma once

#include <optional>
#include <string>
#include <util/filesystem.h>

/// Transforms GCC nested function definitions in a C source file into
/// standard C by lambda-lifting them to file scope. Captured variables
/// from enclosing scopes become explicit pointer parameters (for direct
/// calls) or file-scope static pointer variables (when the nested
/// function is used as a function pointer).
///
/// Returns nullopt if no nested functions are detected (fast no-op).
/// Otherwise returns a tmp_file whose path() contains the transformed
/// source.  The caller should pass that path to Clang instead of the
/// original.
std::optional<file_operations::tmp_file>
transform_nested_functions(const std::string &source_path);
