#define PAGE_SIZE ((1) << XLAT_GRANULARITY_SIZE_SHIFT)

struct rmm_core_manifest {
  int a;
};

int main() {
  static unsigned char el3_rmm_shared_buffer[PAGE_SIZE]
      __attribute__((__aligned__(PAGE_SIZE)));
  static struct rmm_core_manifest *boot_manifest =
      (struct rmm_core_manifest *)el3_rmm_shared_buffer;

  return 0;
}
