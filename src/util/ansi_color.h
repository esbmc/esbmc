/*
    Use ANSI 8/16 codes to output coloured text 
    to the terminal in Linux/Mac/Windows.
    NOTE: The colors can vary depending of the terminal configuration.
    See https://misc.flogisoft.com/bash/tip_colors_and_formatting
*/

#define ansi_red "\033[31m"    // Error, Failed
#define ansi_yellow "\033[33m" // Warning
#define ansi_blue "\033[34m"   // Status, Result
#define ansi_clr                                                               \
  "\033[0m" //  removes all attributes (formatting and colors), add it at the end of each colored text
