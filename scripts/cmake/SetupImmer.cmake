# Provides the header-only immer library as the `immer` interface target.
#
# immer::vector is a persistent (immutable, structurally-shared) bit-mapped
# vector trie: O(1) copy, O(log32 N) index/append, O(N) iteration. guard2tc's
# conjunct list uses it so guard copies (constant in symex) are O(1) instead
# of O(N) deep vector copies — the residual quadratic at deep loop unwinding.
#
# Resolution order mirrors the other downloaded deps:
#   1. With DOWNLOAD_DEPENDENCIES, fetch immer via FetchContent (header-only,
#      so we only need the headers — no build/install step).
#   2. Otherwise locate a system immer via find_package.

if(DOWNLOAD_DEPENDENCIES)
   include(FetchContent)
   fetchcontent_declare(immer
      GIT_REPOSITORY https://github.com/arximboldi/immer.git
      GIT_TAG v0.9.1)
   # Header-only: populate without configuring immer's own CMake (which would
   # pull in its tests/benchmarks/examples).
   fetchcontent_getproperties(immer)
   if(NOT immer_POPULATED)
      fetchcontent_populate(immer)
   endif()
   add_library(immer INTERFACE)
   target_include_directories(immer SYSTEM INTERFACE ${immer_SOURCE_DIR})
   message(STATUS "[immer] using downloaded immer (header-only)")
else()
   find_package(Immer QUIET)
   # immer's installed CMake config has exported the target under different
   # names across versions (bare `immer` and namespaced `immer::immer`).
   # Accept either and alias to the bare `immer` name irep2 links against.
   if(TARGET immer::immer AND NOT TARGET immer)
      add_library(immer ALIAS immer::immer)
   endif()
   if(NOT TARGET immer)
      message(FATAL_ERROR
         "immer not found and DOWNLOAD_DEPENDENCIES is OFF. Install immer "
         "(libimmer-dev / brew install immer) or configure with "
         "-DDOWNLOAD_DEPENDENCIES=On.")
   endif()
   message(STATUS "[immer] using system immer")
endif()
