/*******************************************************************\

Module: Message System. This system is used to send messages through
    ESBMC execution.
Author: Daniel Kroening, kroening@kroening.com

Maintainers:
- @2021: Rafael Sá Menezes, rafael.sa.menezes@outlook.com

\*******************************************************************/
#pragma once

#include <cstdio>
#include <fmt/format.h>
#include <util/message/format.h>
#include <util/location.h>
#include <ctime>
#include <cstring>

/**
 * @brief Verbosity refers to the max level
 * of which inputs are going to be printed out
 *
 * The level adds up to the greater level which means
 * that if the level is set to 3 all messages of value
 * 0,1,2,3 are going to be printed but 4+ will not be printed
 *
 * The number is where it appeared in the definition, in the
 * implementation below Debug is the highest value
 */
enum class VerbosityLevel : char
{
  None,     // No message output
  Error,    // fatal errors are printed
  Warning,  // warnings are printend
  Result,   // results of the analysis (including CE)
  Progress, // progress notifications
  Status,   // all kinds of things esbmc is doing that may be useful to the user
  Debug     // messages that are only useful if you need to debug.
};

struct messaget
{
  static inline class
  {
    template <typename... Args>
    static void println(FILE *f, VerbosityLevel lvl, Args &&...args)
    {
    	     
      	/** 
		* Newly added code to address issue #886.
			* It prepends a well formatted timestamp, file path, function name, line number and log level to each LOG message. 
			* It is transparent to the existing logging function, i.e., log_message.
			* TODO: It can be implemented using a separate function and invoked here and/or elsewhere when required. 		
		*/		
		//========Start===============================================================================================
		struct tm* time_struct; //A time structure consisting of 9 members, such as tm_sec, tm_min, tm_hour, tm_mday, tm_mon, tm_year, etc. 
		char time_info[20]; //A variable to hold the date and time values.
		char log_level[12]; //A variable to hold the log level.
		
		time_t current_time = time(0);
		time_struct = localtime(&current_time);
		strftime (time_info, sizeof(time_info), "%Y-%m-%d %H:%M:%S", time_struct); //"strftime" is to format the date/time structure. 
	   
		switch(lvl){//To prepend the respective log level to each log message.  
			case VerbosityLevel::Error:
				std::strcpy(log_level, "[ERROR]:"); 
				break;
			case VerbosityLevel::Result:
				std::strcpy(log_level, "[RESULT]:");
				break;
			case VerbosityLevel::Warning:
				std::strcpy(log_level, "[WARNING]:");
				break;
			case VerbosityLevel::Progress:
				std::strcpy(log_level, "[PROGRESS]:");
				break;
			case VerbosityLevel::Status:
				std::strcpy(log_level, "[STATUS]:");
				break;
			default:
				std::strcpy(log_level, "[DEBUG]:");
		}
		
		std::fprintf(f, "%s [%s] [%s:%d] %s ", time_info, __FILE__, __FUNCTION__, __LINE__, log_level); /* Prepends the timestamp, file name, file path, 
		* function name, line number and log level to each log message. */
		fmt::print(f, std::forward<Args>(args)...);
	  	fmt::print(f, "\n");
	  	//========End=================================================================================================     
    }

  public:
    VerbosityLevel verbosity;
    FILE *out;
    FILE *err;

    FILE *target(VerbosityLevel lvl) const
    {
      return lvl > verbosity                ? nullptr
             : lvl == VerbosityLevel::Error ? err
                                            : out;
    }

    void set_flushln() const
    {
/* Win32 interprets _IOLBF as _IOFBF (and then chokes on size=0) */
#if !defined(_WIN32) || defined(_WIN64) || defined(__CYGWIN__)
      setvbuf(out, NULL, _IOLBF, 0);
      setvbuf(err, NULL, _IOLBF, 0);
#endif
    }

    template <typename... Args>
    bool logln(VerbosityLevel lvl, Args &&...args) const
    {
      FILE *f = target(lvl);
      if(!f)
        return false;
      println(f, lvl, std::forward<Args>(args)...);
      return true;
    }
  } state = {VerbosityLevel::Status, stdout, stderr};
};

static inline void
print(VerbosityLevel lvl, std::string_view msg, const locationt &)
{
  messaget::state.logln(lvl, "{}", msg);
}

// Macro to generate log functions
#define log_message(name, verbosity)                                           \
  template <typename... Args>                                                  \
  static inline void log_##name(std::string_view fmt, Args &&...args)          \
  {                                                                            \
    messaget::state.logln(verbosity, fmt, std::forward<Args>(args)...);        \
  }

log_message(error, VerbosityLevel::Error);
log_message(result, VerbosityLevel::Result);
log_message(warning, VerbosityLevel::Warning);
log_message(progress, VerbosityLevel::Progress);
log_message(status, VerbosityLevel::Status);
log_message(debug, VerbosityLevel::Debug);

#undef log_message

// TODO: Eventually this will be removed
#ifdef ENABLE_OLD_FRONTEND
#define err_location(E) (E).location().dump()
#endif
