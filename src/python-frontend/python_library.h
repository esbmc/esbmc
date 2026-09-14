#pragma once

class contextt;
class goto_functionst;

void add_cpython_library(contextt &context);

/// Attach the blob's model bodies to \p dest. Runs after goto_convert, which
/// creates their bodyless entries from the declarations.
void link_cpython_library_bodies(goto_functionst &dest);

/// The blob's model bodies. Their symbols carry nil values, so a dependency
/// walk over the context alone cannot see what a model calls.
const goto_functionst &cpython_library_bodies();
