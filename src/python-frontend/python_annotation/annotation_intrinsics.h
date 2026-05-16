#pragma once

#include <map>
#include <string>

namespace python_annotation_intrinsics
{

// Mapping from a Python built-in / std-library identifier to the
// type-annotation string that the Python frontend's annotation pass
// substitutes for it during inference.
//
// The table is held by a function returning a reference to a
// function-local `static const` map so there is exactly one definition
// across all translation units. The data and the keys must be kept
// byte-identical to the original namespace-scope `builtin_functions`
// constant that previously lived in python_annotation.h — this is a
// move, not a rewrite, and any divergence here is a semantic change
// to type inference.
const std::map<std::string, std::string> &builtin_functions();

} // namespace python_annotation_intrinsics
