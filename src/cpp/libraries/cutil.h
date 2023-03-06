#ifndef _CUTIL_H_
#define _CUTIL_H_

#ifdef _WIN32
#   pragma warning( disable : 4996 ) // disable deprecated warning
#endif

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

    // helper typedefs for building DLL
#ifdef _WIN32
#  ifdef BUILD_DLL
#    define DLL_MAPPING  __declspec(dllexport)
#  else
#    define DLL_MAPPING  __declspec(dllimport)
#  endif
#else
#  define DLL_MAPPING
#endif

#ifdef _WIN32
    #define CUTIL_API __stdcall
#else
    #define CUTIL_API
#endif

    ////////////////////////////////////////////////////////////////////////////
    //! CUT bool type
    ////////////////////////////////////////////////////////////////////////////
    enum CUTBoolean
    {
        CUTFalse = 0,
        CUTTrue = 1
    };

    typedef enum CUTBoolean CUTBoolean_t;

# define CUT_CHECK_ERROR(call);

# define CUT_DEVICE_INIT(ARGC, ARGV);

# define  CUT_CHECK_ERROR(call);

# define  CUT_SAFE_MALLOC(call);

# define  CHECK_ERROR(call);

#  define CUDA_SAFE_CALL_NO_SYNC(call) {                                    \
    cudaError err = call;                                                    \
    if( CUDA_SUCCESS != err) {                                                \
/*        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );            */  \
        exit(EXIT_FAILURE);                                                  \
    } }

#  define CUDA_SAFE_CALL( call) CUDA_SAFE_CALL_NO_SYNC(call);                                            \


#  define CUT_SAFE_CALL( call)                                               \
    if( CUTTrue != call) {                                                   \
/*        fprintf(stderr, "Cut error in file '%s' in line %i.\n",              \
                __FILE__, __LINE__);                                     */    \
        exit(EXIT_FAILURE);                                                  \
    }

    ////////////////////////////////////////////////////////////////////////////
    //! Check if command line argument \a flag-name is given
    //! @return CUTTrue if command line argument \a flag_name has been given,
    //!         otherwise 0
    //! @param argc  argc as passed to main()
    //! @param argv  argv as passed to main()
    //! @param flag_name  name of command line flag
    ////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
    CUTBoolean CUTIL_API
    cutCheckCmdLineFlag( const int argc, const char** argv,
                         const char* flag_name);



    ////////////////////////////////////////////////////////////////////////////
    //! Timer functionality

    ////////////////////////////////////////////////////////////////////////////
    //! Create a new timer
    //! @return CUTTrue if a time has been created, otherwise false
    //! @param  name of the new timer, 0 if the creation failed
    ////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
    CUTBoolean CUTIL_API
    cutCreateTimer( unsigned int* name);

    CUTBoolean cutCreateTimer( unsigned int* name){
    	CUTBoolean out = CUTTrue;
    	return out;
    }

    ////////////////////////////////////////////////////////////////////////////
    //! Delete a timer
    //! @return CUTTrue if a time has been deleted, otherwise false
    //! @param  name of the timer to delete
    ////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
    CUTBoolean CUTIL_API
    cutDeleteTimer( unsigned int name);

    CUTBoolean cutDeleteTimer( unsigned int name){
    	CUTBoolean out = CUTTrue;
    	return out;
    }

    ////////////////////////////////////////////////////////////////////////////
    //! Start the time with name \a name
    //! @param name  name of the timer to start
    ////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
    CUTBoolean CUTIL_API
    cutStartTimer( const unsigned int name);

    CUTBoolean cutStartTimer( unsigned int name){
    	CUTBoolean out = CUTTrue;
    	return out;
    }

    ////////////////////////////////////////////////////////////////////////////
    //! Stop the time with name \a name. Does not reset.
    //! @param name  name of the timer to stop
    ////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
    CUTBoolean CUTIL_API
    cutStopTimer( const unsigned int name);

    CUTBoolean cutStopTimer( unsigned int name){
    	CUTBoolean out = CUTTrue;
    	return out;
    }

    ////////////////////////////////////////////////////////////////////////////
    //! Resets the timer's counter.
    //! @param name  name of the timer to reset.
    ////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
    CUTBoolean CUTIL_API
    cutResetTimer( const unsigned int name);

    CUTBoolean cutResetTimer( unsigned int name){
    	CUTBoolean out = CUTTrue;
    	return out;
    }

    ////////////////////////////////////////////////////////////////////////////
    //! Returns total execution time in milliseconds for the timer over all
    //! runs since the last reset or timer creation.
    //! @param name  name of the timer to return the time of
    ////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
    float CUTIL_API
    cutGetTimerValue( const unsigned int name);

    float GetTimerValue( unsigned int name){
    	float out = 1;
    	return out;
    }

    ////////////////////////////////////////////////////////////////////////////
    //! Return the average time in milliseconds for timer execution as the
    //! total  time for the timer dividied by the number of completed (stopped)
    //! runs the timer has made.
    //! Excludes the current running time if the timer is currently running.
    //! @param name  name of the timer to return the time of
    ////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
    float CUTIL_API
    cutGetAverageTimerValue( const unsigned int name);

    ////////////////////////////////////////////////////////////////////////////
    //! Macros


#define CUT_EXIT(argc, argv)                                                 \
    if (!cutCheckCmdLineFlag(argc, (const char**)argv, "noprompt")) {        \
        printf("\nPress ENTER to exit...\n");                                \
        fflush( stdout);                                                     \
/*        fflush( stderr);                                                  */   \
        getchar();                                                           \
    }                                                                        \
    exit(EXIT_SUCCESS);


#ifdef __cplusplus
}
#endif  // #ifdef _DEBUG (else branch)

#endif  // #ifndef _CUTIL_H_
