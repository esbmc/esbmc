/* Test: assigns compliance with ctx->field — PASS
 * Function update_count declares assigns(ctx->count) and only writes ctx->count.
 * Field ctx->capacity is NOT modified, so compliance should pass.
 */

typedef struct {
    int count;
    int capacity;
} queue_t;

void update_count(queue_t *ctx)
{
    __ESBMC_assigns(ctx->count);

    ctx->count = ctx->count + 1;
    /* ctx->capacity is NOT modified — compliance should pass */
}

int main()
{
    queue_t q;
    q.count = 0;
    q.capacity = 10;
    update_count(&q);
    return 0;
}
