# Activates ccache on build

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
else()
  message(AUTHOR_WARNING "ccache not found, incremental builds will be slower")
endif(CCACHE_FOUND)