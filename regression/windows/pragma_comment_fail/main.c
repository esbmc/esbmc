#include <assert.h>

#if (defined(_M_IX86) && !defined(_M_HYBRID_X86_ARM64)) ||                     \
  (defined(_M_X64) && !defined(_M_ARM64EC))
extern int _Avx2WmemEnabled;
__declspec(selectany) int _Avx2WmemEnabledWeakValue = 44;
#  if defined(_M_IX86)
#    pragma comment(                                                           \
      linker, "/alternatename:__Avx2WmemEnabled=__Avx2WmemEnabledWeakValue")
#  else
#    pragma comment(                                                           \
      linker, "/alternatename:_Avx2WmemEnabled=_Avx2WmemEnabledWeakValue")
#  endif
#endif

int main()
{
#if (defined(_M_IX86) && !defined(_M_HYBRID_X86_ARM64)) ||                     \
  (defined(_M_X64) && !defined(_M_ARM64EC))
  assert(_Avx2WmemEnabled == 4444); // should be 44
#endif
}