#pragma once

class contextt;
class goto_functionst;

void add_cpython_library(contextt &context);

/// Attach the blob's model bodies to \p dest. Runs after goto_convert, which
/// creates their bodyless entries from the declarations.
void link_cpython_library_bodies(goto_functionst &dest);
