include (FindPkgConfig)
pkg_search_module (IBEX REQUIRED ibex)
message(STATUS "Found Ibex version ${IBEX_VERSION}")