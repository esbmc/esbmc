#pragma once

class contextt;
class goto_functionst;

void add_cpython_library(contextt &context);

/// Move the model bodies add_cpython_library read out of the blob into
/// \p dest. Must run after goto_convert, which creates the (bodyless) entries
/// for the declarations the blob's symbol table carries.
void link_cpython_library_bodies(goto_functionst &dest);
