/* Test: assigns compliance with empty assigns (pure function)
 * Function pure_read has an empty assigns clause — it must not modify anything.
 */

int g = 10;

int pure_read(void)
{
    __ESBMC_assigns();
    __ESBMC_ensures(__ESBMC_return_value == g);

    return g;
}
