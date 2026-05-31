# Module to provide the mimalloc allocator target `MIMALLOC_TARGET`.
#
# symex is allocation-heavy (millions of small, short-lived irep2 nodes);
# linking mimalloc in place of the system allocator measured ~15% faster
# symex on high-unwind runs. Enabled with -DENABLE_MIMALLOC=On.
#
# Resolution order:
#   1. With DOWNLOAD_DEPENDENCIES, fetch + build mimalloc as a static lib via
#      FetchContent (same mechanism as the other downloaded deps, so it is
#      populated before FetchContent is frozen for the rest of the build).
#   2. Otherwise locate a system/prebuilt mimalloc via find_package.
# MIMALLOC_TARGET is set to the library esbmc should link.

if(NOT ENABLE_MIMALLOC)
   return()
endif()

# Linux only. Static mimalloc with MI_OVERRIDE cannot reliably intercept every
# allocation on macOS (the system zone allocator and libc++/dyld/Obj-C runtime
# allocate before mimalloc's constructors run); a later free() of one of those
# pointers trips "mi_free: pointer does not point to a valid heap space" and
# aborts. The measured symex win (~13%) is a Linux result anyway, so confine
# the override to Linux and leave other platforms on the system allocator.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
   message(STATUS
      "[mimalloc] disabled on ${CMAKE_SYSTEM_NAME}: static MI_OVERRIDE is only "
      "sound on Linux. Using the system allocator.")
   set(ENABLE_MIMALLOC OFF)
   return()
endif()

if(DOWNLOAD_DEPENDENCIES)
   include(FetchContent)
   # mimalloc build knobs: only the static lib, no tests/objects, and
   # MI_OVERRIDE so it replaces malloc/operator new for whatever links it.
   set(MI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
   set(MI_BUILD_OBJECT OFF CACHE BOOL "" FORCE)
   set(MI_BUILD_SHARED OFF CACHE BOOL "" FORCE)
   set(MI_BUILD_STATIC ON CACHE BOOL "" FORCE)
   set(MI_OVERRIDE ON CACHE BOOL "" FORCE)
   fetchcontent_declare(mimalloc
      GIT_REPOSITORY https://github.com/microsoft/mimalloc.git
      GIT_TAG v2.1.7)
   fetchcontent_makeavailable(mimalloc)
   set(MIMALLOC_TARGET mimalloc-static)
   message(STATUS "[mimalloc] using downloaded mimalloc-static")
else()
   # No download requested: use a system mimalloc if present. Default-ON, so
   # do NOT hard-fail when it is missing — just build without it (the global
   # allocator is used instead). Pass -DDOWNLOAD_DEPENDENCIES=On to fetch it.
   find_package(mimalloc QUIET)
   if(TARGET mimalloc-static)
      set(MIMALLOC_TARGET mimalloc-static)
   elseif(TARGET mimalloc)
      set(MIMALLOC_TARGET mimalloc)
   else()
      message(STATUS
         "[mimalloc] ENABLE_MIMALLOC is ON but no system mimalloc was found "
         "and DOWNLOAD_DEPENDENCIES is OFF; building without mimalloc. "
         "Install mimalloc or configure with -DDOWNLOAD_DEPENDENCIES=On.")
      set(ENABLE_MIMALLOC OFF)
      return()
   endif()
   message(STATUS "[mimalloc] using system mimalloc (${MIMALLOC_TARGET})")
endif()
