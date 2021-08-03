extern void abort(void);

extern void __assert_fail (const char *__assertion, const char *__file,
      unsigned int __line, const char *__function)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__noreturn__));
extern void __assert_perror_fail (int __errnum, const char *__file,
      unsigned int __line, const char *__function)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__noreturn__));
extern void __assert (const char *__assertion, const char *__file, int __line)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__noreturn__));

void reach_error() { ((void) sizeof ((0) ? 1 : 0), __extension__ ({ if (0) ; else __assert_fail ("0", "43_2a_bitvector_linux-3.16-rc1.tar.xz-43_2a-drivers--usb--host--max3421-hcd.ko-entry_point.cil.out.c", 3, __extension__ __PRETTY_FUNCTION__); })); }
struct kernel_symbol {
   unsigned long value ;
   char const *name ;
};
struct module;
typedef signed char __s8;
typedef unsigned char __u8;
typedef short __s16;
typedef unsigned short __u16;
typedef int __s32;
typedef unsigned int __u32;
typedef unsigned long long __u64;
typedef unsigned char u8;
typedef short s16;
typedef unsigned short u16;
typedef int s32;
typedef unsigned int u32;
typedef long long s64;
typedef unsigned long long u64;
typedef long __kernel_long_t;
typedef unsigned long __kernel_ulong_t;
typedef int __kernel_pid_t;
typedef unsigned int __kernel_uid32_t;
typedef unsigned int __kernel_gid32_t;
typedef __kernel_ulong_t __kernel_size_t;
typedef __kernel_long_t __kernel_ssize_t;
typedef long long __kernel_loff_t;
typedef __kernel_long_t __kernel_time_t;
typedef __kernel_long_t __kernel_clock_t;
typedef int __kernel_timer_t;
typedef int __kernel_clockid_t;
typedef __u16 __le16;
typedef __u32 __le32;
typedef __u32 __kernel_dev_t;
typedef __kernel_dev_t dev_t;
typedef unsigned short umode_t;
typedef __kernel_pid_t pid_t;
typedef __kernel_clockid_t clockid_t;
typedef _Bool bool;
typedef __kernel_uid32_t uid_t;
typedef __kernel_gid32_t gid_t;
typedef __kernel_loff_t loff_t;
typedef __kernel_size_t size_t;
typedef __kernel_ssize_t ssize_t;
typedef __kernel_time_t time_t;
typedef __s32 int32_t;
typedef __u32 uint32_t;
typedef unsigned long sector_t;
typedef unsigned long blkcnt_t;
typedef u64 dma_addr_t;
typedef unsigned int gfp_t;
typedef unsigned int fmode_t;
typedef unsigned int oom_flags_t;
typedef u64 phys_addr_t;
typedef phys_addr_t resource_size_t;
struct __anonstruct_atomic_t_6 {
   int counter ;
};
typedef struct __anonstruct_atomic_t_6 atomic_t;
struct __anonstruct_atomic64_t_7 {
   long counter ;
};
typedef struct __anonstruct_atomic64_t_7 atomic64_t;
struct list_head {
   struct list_head *next ;
   struct list_head *prev ;
};
struct hlist_node;
struct hlist_head {
   struct hlist_node *first ;
};
struct hlist_node {
   struct hlist_node *next ;
   struct hlist_node **pprev ;
};
struct callback_head {
   struct callback_head *next ;
   void (*func)(struct callback_head * ) ;
};
struct pt_regs {
   unsigned long r15 ;
   unsigned long r14 ;
   unsigned long r13 ;
   unsigned long r12 ;
   unsigned long bp ;
   unsigned long bx ;
   unsigned long r11 ;
   unsigned long r10 ;
   unsigned long r9 ;
   unsigned long r8 ;
   unsigned long ax ;
   unsigned long cx ;
   unsigned long dx ;
   unsigned long si ;
   unsigned long di ;
   unsigned long orig_ax ;
   unsigned long ip ;
   unsigned long cs ;
   unsigned long flags ;
   unsigned long sp ;
   unsigned long ss ;
};
struct __anonstruct_ldv_1022_9 {
   unsigned int a ;
   unsigned int b ;
};
struct __anonstruct_ldv_1037_10 {
   u16 limit0 ;
   u16 base0 ;
   unsigned char base1 ;
   unsigned char type : 4 ;
   unsigned char s : 1 ;
   unsigned char dpl : 2 ;
   unsigned char p : 1 ;
   unsigned char limit : 4 ;
   unsigned char avl : 1 ;
   unsigned char l : 1 ;
   unsigned char d : 1 ;
   unsigned char g : 1 ;
   unsigned char base2 ;
};
union __anonunion_ldv_1038_8 {
   struct __anonstruct_ldv_1022_9 ldv_1022 ;
   struct __anonstruct_ldv_1037_10 ldv_1037 ;
};
struct desc_struct {
   union __anonunion_ldv_1038_8 ldv_1038 ;
};
typedef unsigned long pteval_t;
typedef unsigned long pgdval_t;
typedef unsigned long pgprotval_t;
struct __anonstruct_pte_t_11 {
   pteval_t pte ;
};
typedef struct __anonstruct_pte_t_11 pte_t;
struct pgprot {
   pgprotval_t pgprot ;
};
typedef struct pgprot pgprot_t;
struct __anonstruct_pgd_t_12 {
   pgdval_t pgd ;
};
typedef struct __anonstruct_pgd_t_12 pgd_t;
struct page;
typedef struct page *pgtable_t;
struct file;
struct seq_file;
struct thread_struct;
struct mm_struct;
struct task_struct;
struct cpumask;
struct arch_spinlock;
typedef u16 __ticket_t;
typedef u32 __ticketpair_t;
struct __raw_tickets {
   __ticket_t head ;
   __ticket_t tail ;
};
union __anonunion_ldv_1458_15 {
   __ticketpair_t head_tail ;
   struct __raw_tickets tickets ;
};
struct arch_spinlock {
   union __anonunion_ldv_1458_15 ldv_1458 ;
};
typedef struct arch_spinlock arch_spinlock_t;
struct qrwlock {
   atomic_t cnts ;
   arch_spinlock_t lock ;
};
typedef struct qrwlock arch_rwlock_t;
typedef void (*ctor_fn_t)(void);
struct _ddebug {
   char const *modname ;
   char const *function ;
   char const *filename ;
   char const *format ;
   unsigned int lineno : 18 ;
   unsigned char flags ;
};
struct device;
struct file_operations;
struct completion;
struct pid;
struct bug_entry {
   int bug_addr_disp ;
   int file_disp ;
   unsigned short line ;
   unsigned short flags ;
};
struct timespec;
struct kernel_vm86_regs {
   struct pt_regs pt ;
   unsigned short es ;
   unsigned short __esh ;
   unsigned short ds ;
   unsigned short __dsh ;
   unsigned short fs ;
   unsigned short __fsh ;
   unsigned short gs ;
   unsigned short __gsh ;
};
union __anonunion_ldv_2998_20 {
   struct pt_regs *regs ;
   struct kernel_vm86_regs *vm86 ;
};
struct math_emu_info {
   long ___orig_eip ;
   union __anonunion_ldv_2998_20 ldv_2998 ;
};
struct cpumask {
   unsigned long bits[128U] ;
};
typedef struct cpumask cpumask_t;
typedef struct cpumask *cpumask_var_t;
struct seq_operations;
struct i387_fsave_struct {
   u32 cwd ;
   u32 swd ;
   u32 twd ;
   u32 fip ;
   u32 fcs ;
   u32 foo ;
   u32 fos ;
   u32 st_space[20U] ;
   u32 status ;
};
struct __anonstruct_ldv_5289_25 {
   u64 rip ;
   u64 rdp ;
};
struct __anonstruct_ldv_5295_26 {
   u32 fip ;
   u32 fcs ;
   u32 foo ;
   u32 fos ;
};
union __anonunion_ldv_5296_24 {
   struct __anonstruct_ldv_5289_25 ldv_5289 ;
   struct __anonstruct_ldv_5295_26 ldv_5295 ;
};
union __anonunion_ldv_5305_27 {
   u32 padding1[12U] ;
   u32 sw_reserved[12U] ;
};
struct i387_fxsave_struct {
   u16 cwd ;
   u16 swd ;
   u16 twd ;
   u16 fop ;
   union __anonunion_ldv_5296_24 ldv_5296 ;
   u32 mxcsr ;
   u32 mxcsr_mask ;
   u32 st_space[32U] ;
   u32 xmm_space[64U] ;
   u32 padding[12U] ;
   union __anonunion_ldv_5305_27 ldv_5305 ;
};
struct i387_soft_struct {
   u32 cwd ;
   u32 swd ;
   u32 twd ;
   u32 fip ;
   u32 fcs ;
   u32 foo ;
   u32 fos ;
   u32 st_space[20U] ;
   u8 ftop ;
   u8 changed ;
   u8 lookahead ;
   u8 no_update ;
   u8 rm ;
   u8 alimit ;
   struct math_emu_info *info ;
   u32 entry_eip ;
};
struct ymmh_struct {
   u32 ymmh_space[64U] ;
};
struct lwp_struct {
   u8 reserved[128U] ;
};
struct bndregs_struct {
   u64 bndregs[8U] ;
};
struct bndcsr_struct {
   u64 cfg_reg_u ;
   u64 status_reg ;
};
struct xsave_hdr_struct {
   u64 xstate_bv ;
   u64 reserved1[2U] ;
   u64 reserved2[5U] ;
};
struct xsave_struct {
   struct i387_fxsave_struct i387 ;
   struct xsave_hdr_struct xsave_hdr ;
   struct ymmh_struct ymmh ;
   struct lwp_struct lwp ;
   struct bndregs_struct bndregs ;
   struct bndcsr_struct bndcsr ;
};
union thread_xstate {
   struct i387_fsave_struct fsave ;
   struct i387_fxsave_struct fxsave ;
   struct i387_soft_struct soft ;
   struct xsave_struct xsave ;
};
struct fpu {
   unsigned int last_cpu ;
   unsigned int has_fpu ;
   union thread_xstate *state ;
};
struct kmem_cache;
struct perf_event;
struct thread_struct {
   struct desc_struct tls_array[3U] ;
   unsigned long sp0 ;
   unsigned long sp ;
   unsigned long usersp ;
   unsigned short es ;
   unsigned short ds ;
   unsigned short fsindex ;
   unsigned short gsindex ;
   unsigned long fs ;
   unsigned long gs ;
   struct perf_event *ptrace_bps[4U] ;
   unsigned long debugreg6 ;
   unsigned long ptrace_dr7 ;
   unsigned long cr2 ;
   unsigned long trap_nr ;
   unsigned long error_code ;
   struct fpu fpu ;
   unsigned long *io_bitmap_ptr ;
   unsigned long iopl ;
   unsigned int io_bitmap_max ;
   unsigned char fpu_counter ;
};
typedef atomic64_t atomic_long_t;
struct lockdep_map;
struct stack_trace {
   unsigned int nr_entries ;
   unsigned int max_entries ;
   unsigned long *entries ;
   int skip ;
};
struct lockdep_subclass_key {
   char __one_byte ;
} __attribute__((__packed__)) ;
struct lock_class_key {
   struct lockdep_subclass_key subkeys[8U] ;
};
struct lock_class {
   struct list_head hash_entry ;
   struct list_head lock_entry ;
   struct lockdep_subclass_key *key ;
   unsigned int subclass ;
   unsigned int dep_gen_id ;
   unsigned long usage_mask ;
   struct stack_trace usage_traces[13U] ;
   struct list_head locks_after ;
   struct list_head locks_before ;
   unsigned int version ;
   unsigned long ops ;
   char const *name ;
   int name_version ;
   unsigned long contention_point[4U] ;
   unsigned long contending_point[4U] ;
};
struct lockdep_map {
   struct lock_class_key *key ;
   struct lock_class *class_cache[2U] ;
   char const *name ;
   int cpu ;
   unsigned long ip ;
};
struct held_lock {
   u64 prev_chain_key ;
   unsigned long acquire_ip ;
   struct lockdep_map *instance ;
   struct lockdep_map *nest_lock ;
   u64 waittime_stamp ;
   u64 holdtime_stamp ;
   unsigned short class_idx : 13 ;
   unsigned char irq_context : 2 ;
   unsigned char trylock : 1 ;
   unsigned char read : 2 ;
   unsigned char check : 1 ;
   unsigned char hardirqs_off : 1 ;
   unsigned short references : 12 ;
};
struct raw_spinlock {
   arch_spinlock_t raw_lock ;
   unsigned int magic ;
   unsigned int owner_cpu ;
   void *owner ;
   struct lockdep_map dep_map ;
};
typedef struct raw_spinlock raw_spinlock_t;
struct __anonstruct_ldv_6346_31 {
   u8 __padding[24U] ;
   struct lockdep_map dep_map ;
};
union __anonunion_ldv_6347_30 {
   struct raw_spinlock rlock ;
   struct __anonstruct_ldv_6346_31 ldv_6346 ;
};
struct spinlock {
   union __anonunion_ldv_6347_30 ldv_6347 ;
};
typedef struct spinlock spinlock_t;
struct __anonstruct_rwlock_t_32 {
   arch_rwlock_t raw_lock ;
   unsigned int magic ;
   unsigned int owner_cpu ;
   void *owner ;
   struct lockdep_map dep_map ;
};
typedef struct __anonstruct_rwlock_t_32 rwlock_t;
struct seqcount {
   unsigned int sequence ;
   struct lockdep_map dep_map ;
};
typedef struct seqcount seqcount_t;
struct timespec {
   __kernel_time_t tv_sec ;
   long tv_nsec ;
};
struct user_namespace;
struct __anonstruct_kuid_t_34 {
   uid_t val ;
};
typedef struct __anonstruct_kuid_t_34 kuid_t;
struct __anonstruct_kgid_t_35 {
   gid_t val ;
};
typedef struct __anonstruct_kgid_t_35 kgid_t;
struct kstat {
   u64 ino ;
   dev_t dev ;
   umode_t mode ;
   unsigned int nlink ;
   kuid_t uid ;
   kgid_t gid ;
   dev_t rdev ;
   loff_t size ;
   struct timespec atime ;
   struct timespec mtime ;
   struct timespec ctime ;
   unsigned long blksize ;
   unsigned long long blocks ;
};
struct __wait_queue_head {
   spinlock_t lock ;
   struct list_head task_list ;
};
typedef struct __wait_queue_head wait_queue_head_t;
struct __anonstruct_nodemask_t_36 {
   unsigned long bits[16U] ;
};
typedef struct __anonstruct_nodemask_t_36 nodemask_t;
struct optimistic_spin_queue;
struct mutex {
   atomic_t count ;
   spinlock_t wait_lock ;
   struct list_head wait_list ;
   struct task_struct *owner ;
   char const *name ;
   void *magic ;
   struct lockdep_map dep_map ;
};
struct mutex_waiter {
   struct list_head list ;
   struct task_struct *task ;
   void *magic ;
};
struct rw_semaphore;
struct rw_semaphore {
   long count ;
   raw_spinlock_t wait_lock ;
   struct list_head wait_list ;
   struct task_struct *owner ;
   struct optimistic_spin_queue *osq ;
   struct lockdep_map dep_map ;
};
struct completion {
   unsigned int done ;
   wait_queue_head_t wait ;
};
struct llist_node;
struct llist_node {
   struct llist_node *next ;
};
union ktime {
   s64 tv64 ;
};
typedef union ktime ktime_t;
struct tvec_base;
struct timer_list {
   struct list_head entry ;
   unsigned long expires ;
   struct tvec_base *base ;
   void (*function)(unsigned long ) ;
   unsigned long data ;
   int slack ;
   int start_pid ;
   void *start_site ;
   char start_comm[16U] ;
   struct lockdep_map lockdep_map ;
};
struct hrtimer;
enum hrtimer_restart;
struct workqueue_struct;
struct work_struct;
struct work_struct {
   atomic_long_t data ;
   struct list_head entry ;
   void (*func)(struct work_struct * ) ;
   struct lockdep_map lockdep_map ;
};
struct delayed_work {
   struct work_struct work ;
   struct timer_list timer ;
   struct workqueue_struct *wq ;
   int cpu ;
};
struct pm_message {
   int event ;
};
typedef struct pm_message pm_message_t;
struct dev_pm_ops {
   int (*prepare)(struct device * ) ;
   void (*complete)(struct device * ) ;
   int (*suspend)(struct device * ) ;
   int (*resume)(struct device * ) ;
   int (*freeze)(struct device * ) ;
   int (*thaw)(struct device * ) ;
   int (*poweroff)(struct device * ) ;
   int (*restore)(struct device * ) ;
   int (*suspend_late)(struct device * ) ;
   int (*resume_early)(struct device * ) ;
   int (*freeze_late)(struct device * ) ;
   int (*thaw_early)(struct device * ) ;
   int (*poweroff_late)(struct device * ) ;
   int (*restore_early)(struct device * ) ;
   int (*suspend_noirq)(struct device * ) ;
   int (*resume_noirq)(struct device * ) ;
   int (*freeze_noirq)(struct device * ) ;
   int (*thaw_noirq)(struct device * ) ;
   int (*poweroff_noirq)(struct device * ) ;
   int (*restore_noirq)(struct device * ) ;
   int (*runtime_suspend)(struct device * ) ;
   int (*runtime_resume)(struct device * ) ;
   int (*runtime_idle)(struct device * ) ;
};
enum rpm_status {
    RPM_ACTIVE = 0,
    RPM_RESUMING = 1,
    RPM_SUSPENDED = 2,
    RPM_SUSPENDING = 3
} ;
enum rpm_request {
    RPM_REQ_NONE = 0,
    RPM_REQ_IDLE = 1,
    RPM_REQ_SUSPEND = 2,
    RPM_REQ_AUTOSUSPEND = 3,
    RPM_REQ_RESUME = 4
} ;
struct wakeup_source;
struct pm_subsys_data {
   spinlock_t lock ;
   unsigned int refcount ;
   struct list_head clock_list ;
};
struct dev_pm_qos;
struct dev_pm_info {
   pm_message_t power_state ;
   unsigned char can_wakeup : 1 ;
   unsigned char async_suspend : 1 ;
   bool is_prepared ;
   bool is_suspended ;
   bool is_noirq_suspended ;
   bool is_late_suspended ;
   bool ignore_children ;
   bool early_init ;
   bool direct_complete ;
   spinlock_t lock ;
   struct list_head entry ;
   struct completion completion ;
   struct wakeup_source *wakeup ;
   bool wakeup_path ;
   bool syscore ;
   struct timer_list suspend_timer ;
   unsigned long timer_expires ;
   struct work_struct work ;
   wait_queue_head_t wait_queue ;
   atomic_t usage_count ;
   atomic_t child_count ;
   unsigned char disable_depth : 3 ;
   unsigned char idle_notification : 1 ;
   unsigned char request_pending : 1 ;
   unsigned char deferred_resume : 1 ;
   unsigned char run_wake : 1 ;
   unsigned char runtime_auto : 1 ;
   unsigned char no_callbacks : 1 ;
   unsigned char irq_safe : 1 ;
   unsigned char use_autosuspend : 1 ;
   unsigned char timer_autosuspends : 1 ;
   unsigned char memalloc_noio : 1 ;
   enum rpm_request request ;
   enum rpm_status runtime_status ;
   int runtime_error ;
   int autosuspend_delay ;
   unsigned long last_busy ;
   unsigned long active_jiffies ;
   unsigned long suspended_jiffies ;
   unsigned long accounting_timestamp ;
   struct pm_subsys_data *subsys_data ;
   void (*set_latency_tolerance)(struct device * , s32 ) ;
   struct dev_pm_qos *qos ;
};
struct dev_pm_domain {
   struct dev_pm_ops ops ;
};
struct __anonstruct_mm_context_t_101 {
   void *ldt ;
   int size ;
   unsigned short ia32_compat ;
   struct mutex lock ;
   void *vdso ;
};
typedef struct __anonstruct_mm_context_t_101 mm_context_t;
struct rb_node {
   unsigned long __rb_parent_color ;
   struct rb_node *rb_right ;
   struct rb_node *rb_left ;
} __attribute__((__aligned__(sizeof(long )))) ;
struct rb_root {
   struct rb_node *rb_node ;
};
struct vm_area_struct;
struct nsproxy;
struct cred;
struct inode;
struct arch_uprobe_task {
   unsigned long saved_scratch_register ;
   unsigned int saved_trap_nr ;
   unsigned int saved_tf ;
};
enum uprobe_task_state {
    UTASK_RUNNING = 0,
    UTASK_SSTEP = 1,
    UTASK_SSTEP_ACK = 2,
    UTASK_SSTEP_TRAPPED = 3
} ;
struct __anonstruct_ldv_14006_136 {
   struct arch_uprobe_task autask ;
   unsigned long vaddr ;
};
struct __anonstruct_ldv_14010_137 {
   struct callback_head dup_xol_work ;
   unsigned long dup_xol_addr ;
};
union __anonunion_ldv_14011_135 {
   struct __anonstruct_ldv_14006_136 ldv_14006 ;
   struct __anonstruct_ldv_14010_137 ldv_14010 ;
};
struct uprobe;
struct return_instance;
struct uprobe_task {
   enum uprobe_task_state state ;
   union __anonunion_ldv_14011_135 ldv_14011 ;
   struct uprobe *active_uprobe ;
   unsigned long xol_vaddr ;
   struct return_instance *return_instances ;
   unsigned int depth ;
};
struct xol_area;
struct uprobes_state {
   struct xol_area *xol_area ;
};
struct address_space;
union __anonunion_ldv_14120_138 {
   struct address_space *mapping ;
   void *s_mem ;
};
union __anonunion_ldv_14126_140 {
   unsigned long index ;
   void *freelist ;
   bool pfmemalloc ;
};
struct __anonstruct_ldv_14136_144 {
   unsigned short inuse ;
   unsigned short objects : 15 ;
   unsigned char frozen : 1 ;
};
union __anonunion_ldv_14138_143 {
   atomic_t _mapcount ;
   struct __anonstruct_ldv_14136_144 ldv_14136 ;
   int units ;
};
struct __anonstruct_ldv_14140_142 {
   union __anonunion_ldv_14138_143 ldv_14138 ;
   atomic_t _count ;
};
union __anonunion_ldv_14142_141 {
   unsigned long counters ;
   struct __anonstruct_ldv_14140_142 ldv_14140 ;
   unsigned int active ;
};
struct __anonstruct_ldv_14143_139 {
   union __anonunion_ldv_14126_140 ldv_14126 ;
   union __anonunion_ldv_14142_141 ldv_14142 ;
};
struct __anonstruct_ldv_14150_146 {
   struct page *next ;
   int pages ;
   int pobjects ;
};
struct slab;
union __anonunion_ldv_14155_145 {
   struct list_head lru ;
   struct __anonstruct_ldv_14150_146 ldv_14150 ;
   struct slab *slab_page ;
   struct callback_head callback_head ;
   pgtable_t pmd_huge_pte ;
};
union __anonunion_ldv_14161_147 {
   unsigned long private ;
   spinlock_t *ptl ;
   struct kmem_cache *slab_cache ;
   struct page *first_page ;
};
struct page {
   unsigned long flags ;
   union __anonunion_ldv_14120_138 ldv_14120 ;
   struct __anonstruct_ldv_14143_139 ldv_14143 ;
   union __anonunion_ldv_14155_145 ldv_14155 ;
   union __anonunion_ldv_14161_147 ldv_14161 ;
   unsigned long debug_flags ;
};
struct page_frag {
   struct page *page ;
   __u32 offset ;
   __u32 size ;
};
struct __anonstruct_linear_149 {
   struct rb_node rb ;
   unsigned long rb_subtree_last ;
};
union __anonunion_shared_148 {
   struct __anonstruct_linear_149 linear ;
   struct list_head nonlinear ;
};
struct anon_vma;
struct vm_operations_struct;
struct mempolicy;
struct vm_area_struct {
   unsigned long vm_start ;
   unsigned long vm_end ;
   struct vm_area_struct *vm_next ;
   struct vm_area_struct *vm_prev ;
   struct rb_node vm_rb ;
   unsigned long rb_subtree_gap ;
   struct mm_struct *vm_mm ;
   pgprot_t vm_page_prot ;
   unsigned long vm_flags ;
   union __anonunion_shared_148 shared ;
   struct list_head anon_vma_chain ;
   struct anon_vma *anon_vma ;
   struct vm_operations_struct const *vm_ops ;
   unsigned long vm_pgoff ;
   struct file *vm_file ;
   void *vm_private_data ;
   struct mempolicy *vm_policy ;
};
struct core_thread {
   struct task_struct *task ;
   struct core_thread *next ;
};
struct core_state {
   atomic_t nr_threads ;
   struct core_thread dumper ;
   struct completion startup ;
};
struct task_rss_stat {
   int events ;
   int count[3U] ;
};
struct mm_rss_stat {
   atomic_long_t count[3U] ;
};
struct kioctx_table;
struct linux_binfmt;
struct mmu_notifier_mm;
struct mm_struct {
   struct vm_area_struct *mmap ;
   struct rb_root mm_rb ;
   u32 vmacache_seqnum ;
   unsigned long (*get_unmapped_area)(struct file * , unsigned long , unsigned long ,
                                      unsigned long , unsigned long ) ;
   unsigned long mmap_base ;
   unsigned long mmap_legacy_base ;
   unsigned long task_size ;
   unsigned long highest_vm_end ;
   pgd_t *pgd ;
   atomic_t mm_users ;
   atomic_t mm_count ;
   atomic_long_t nr_ptes ;
   int map_count ;
   spinlock_t page_table_lock ;
   struct rw_semaphore mmap_sem ;
   struct list_head mmlist ;
   unsigned long hiwater_rss ;
   unsigned long hiwater_vm ;
   unsigned long total_vm ;
   unsigned long locked_vm ;
   unsigned long pinned_vm ;
   unsigned long shared_vm ;
   unsigned long exec_vm ;
   unsigned long stack_vm ;
   unsigned long def_flags ;
   unsigned long start_code ;
   unsigned long end_code ;
   unsigned long start_data ;
   unsigned long end_data ;
   unsigned long start_brk ;
   unsigned long brk ;
   unsigned long start_stack ;
   unsigned long arg_start ;
   unsigned long arg_end ;
   unsigned long env_start ;
   unsigned long env_end ;
   unsigned long saved_auxv[46U] ;
   struct mm_rss_stat rss_stat ;
   struct linux_binfmt *binfmt ;
   cpumask_var_t cpu_vm_mask_var ;
   mm_context_t context ;
   unsigned long flags ;
   struct core_state *core_state ;
   spinlock_t ioctx_lock ;
   struct kioctx_table *ioctx_table ;
   struct task_struct *owner ;
   struct file *exe_file ;
   struct mmu_notifier_mm *mmu_notifier_mm ;
   struct cpumask cpumask_allocation ;
   unsigned long numa_next_scan ;
   unsigned long numa_scan_offset ;
   int numa_scan_seq ;
   bool tlb_flush_pending ;
   struct uprobes_state uprobes_state ;
};
typedef __u64 Elf64_Addr;
typedef __u16 Elf64_Half;
typedef __u32 Elf64_Word;
typedef __u64 Elf64_Xword;
struct elf64_sym {
   Elf64_Word st_name ;
   unsigned char st_info ;
   unsigned char st_other ;
   Elf64_Half st_shndx ;
   Elf64_Addr st_value ;
   Elf64_Xword st_size ;
};
typedef struct elf64_sym Elf64_Sym;
union __anonunion_ldv_14524_153 {
   unsigned long bitmap[4U] ;
   struct callback_head callback_head ;
};
struct idr_layer {
   int prefix ;
   int layer ;
   struct idr_layer *ary[256U] ;
   int count ;
   union __anonunion_ldv_14524_153 ldv_14524 ;
};
struct idr {
   struct idr_layer *hint ;
   struct idr_layer *top ;
   int layers ;
   int cur ;
   spinlock_t lock ;
   int id_free_cnt ;
   struct idr_layer *id_free ;
};
struct ida_bitmap {
   long nr_busy ;
   unsigned long bitmap[15U] ;
};
struct ida {
   struct idr idr ;
   struct ida_bitmap *free_bitmap ;
};
struct dentry;
struct iattr;
struct super_block;
struct file_system_type;
struct kernfs_open_node;
struct kernfs_iattrs;
struct kernfs_root;
struct kernfs_elem_dir {
   unsigned long subdirs ;
   struct rb_root children ;
   struct kernfs_root *root ;
};
struct kernfs_node;
struct kernfs_elem_symlink {
   struct kernfs_node *target_kn ;
};
struct kernfs_ops;
struct kernfs_elem_attr {
   struct kernfs_ops const *ops ;
   struct kernfs_open_node *open ;
   loff_t size ;
};
union __anonunion_ldv_14668_154 {
   struct kernfs_elem_dir dir ;
   struct kernfs_elem_symlink symlink ;
   struct kernfs_elem_attr attr ;
};
struct kernfs_node {
   atomic_t count ;
   atomic_t active ;
   struct lockdep_map dep_map ;
   struct kernfs_node *parent ;
   char const *name ;
   struct rb_node rb ;
   void const *ns ;
   unsigned int hash ;
   union __anonunion_ldv_14668_154 ldv_14668 ;
   void *priv ;
   unsigned short flags ;
   umode_t mode ;
   unsigned int ino ;
   struct kernfs_iattrs *iattr ;
};
struct kernfs_syscall_ops {
   int (*remount_fs)(struct kernfs_root * , int * , char * ) ;
   int (*show_options)(struct seq_file * , struct kernfs_root * ) ;
   int (*mkdir)(struct kernfs_node * , char const * , umode_t ) ;
   int (*rmdir)(struct kernfs_node * ) ;
   int (*rename)(struct kernfs_node * , struct kernfs_node * , char const * ) ;
};
struct kernfs_root {
   struct kernfs_node *kn ;
   unsigned int flags ;
   struct ida ino_ida ;
   struct kernfs_syscall_ops *syscall_ops ;
   struct list_head supers ;
   wait_queue_head_t deactivate_waitq ;
};
struct kernfs_open_file {
   struct kernfs_node *kn ;
   struct file *file ;
   void *priv ;
   struct mutex mutex ;
   int event ;
   struct list_head list ;
   size_t atomic_write_len ;
   bool mmapped ;
   struct vm_operations_struct const *vm_ops ;
};
struct kernfs_ops {
   int (*seq_show)(struct seq_file * , void * ) ;
   void *(*seq_start)(struct seq_file * , loff_t * ) ;
   void *(*seq_next)(struct seq_file * , void * , loff_t * ) ;
   void (*seq_stop)(struct seq_file * , void * ) ;
   ssize_t (*read)(struct kernfs_open_file * , char * , size_t , loff_t ) ;
   size_t atomic_write_len ;
   ssize_t (*write)(struct kernfs_open_file * , char * , size_t , loff_t ) ;
   int (*mmap)(struct kernfs_open_file * , struct vm_area_struct * ) ;
   struct lock_class_key lockdep_key ;
};
struct sock;
struct kobject;
enum kobj_ns_type {
    KOBJ_NS_TYPE_NONE = 0,
    KOBJ_NS_TYPE_NET = 1,
    KOBJ_NS_TYPES = 2
} ;
struct kobj_ns_type_operations {
   enum kobj_ns_type type ;
   bool (*current_may_mount)(void) ;
   void *(*grab_current_ns)(void) ;
   void const *(*netlink_ns)(struct sock * ) ;
   void const *(*initial_ns)(void) ;
   void (*drop_ns)(void * ) ;
};
struct bin_attribute;
struct attribute {
   char const *name ;
   umode_t mode ;
   bool ignore_lockdep ;
   struct lock_class_key *key ;
   struct lock_class_key skey ;
};
struct attribute_group {
   char const *name ;
   umode_t (*is_visible)(struct kobject * , struct attribute * , int ) ;
   struct attribute **attrs ;
   struct bin_attribute **bin_attrs ;
};
struct bin_attribute {
   struct attribute attr ;
   size_t size ;
   void *private ;
   ssize_t (*read)(struct file * , struct kobject * , struct bin_attribute * , char * ,
                   loff_t , size_t ) ;
   ssize_t (*write)(struct file * , struct kobject * , struct bin_attribute * , char * ,
                    loff_t , size_t ) ;
   int (*mmap)(struct file * , struct kobject * , struct bin_attribute * , struct vm_area_struct * ) ;
};
struct sysfs_ops {
   ssize_t (*show)(struct kobject * , struct attribute * , char * ) ;
   ssize_t (*store)(struct kobject * , struct attribute * , char const * , size_t ) ;
};
struct kref {
   atomic_t refcount ;
};
struct kset;
struct kobj_type;
struct kobject {
   char const *name ;
   struct list_head entry ;
   struct kobject *parent ;
   struct kset *kset ;
   struct kobj_type *ktype ;
   struct kernfs_node *sd ;
   struct kref kref ;
   struct delayed_work release ;
   unsigned char state_initialized : 1 ;
   unsigned char state_in_sysfs : 1 ;
   unsigned char state_add_uevent_sent : 1 ;
   unsigned char state_remove_uevent_sent : 1 ;
   unsigned char uevent_suppress : 1 ;
};
struct kobj_type {
   void (*release)(struct kobject * ) ;
   struct sysfs_ops const *sysfs_ops ;
   struct attribute **default_attrs ;
   struct kobj_ns_type_operations const *(*child_ns_type)(struct kobject * ) ;
   void const *(*namespace)(struct kobject * ) ;
};
struct kobj_uevent_env {
   char *argv[3U] ;
   char *envp[32U] ;
   int envp_idx ;
   char buf[2048U] ;
   int buflen ;
};
struct kset_uevent_ops {
   int (* const filter)(struct kset * , struct kobject * ) ;
   char const *(* const name)(struct kset * , struct kobject * ) ;
   int (* const uevent)(struct kset * , struct kobject * , struct kobj_uevent_env * ) ;
};
struct kset {
   struct list_head list ;
   spinlock_t list_lock ;
   struct kobject kobj ;
   struct kset_uevent_ops const *uevent_ops ;
};
struct kernel_param;
struct kernel_param_ops {
   unsigned int flags ;
   int (*set)(char const * , struct kernel_param const * ) ;
   int (*get)(char * , struct kernel_param const * ) ;
   void (*free)(void * ) ;
};
struct kparam_string;
struct kparam_array;
union __anonunion_ldv_15343_155 {
   void *arg ;
   struct kparam_string const *str ;
   struct kparam_array const *arr ;
};
struct kernel_param {
   char const *name ;
   struct kernel_param_ops const *ops ;
   u16 perm ;
   s16 level ;
   union __anonunion_ldv_15343_155 ldv_15343 ;
};
struct kparam_string {
   unsigned int maxlen ;
   char *string ;
};
struct kparam_array {
   unsigned int max ;
   unsigned int elemsize ;
   unsigned int *num ;
   struct kernel_param_ops const *ops ;
   void *elem ;
};
struct mod_arch_specific {
};
struct module_param_attrs;
struct module_kobject {
   struct kobject kobj ;
   struct module *mod ;
   struct kobject *drivers_dir ;
   struct module_param_attrs *mp ;
   struct completion *kobj_completion ;
};
struct module_attribute {
   struct attribute attr ;
   ssize_t (*show)(struct module_attribute * , struct module_kobject * , char * ) ;
   ssize_t (*store)(struct module_attribute * , struct module_kobject * , char const * ,
                    size_t ) ;
   void (*setup)(struct module * , char const * ) ;
   int (*test)(struct module * ) ;
   void (*free)(struct module * ) ;
};
struct exception_table_entry;
enum module_state {
    MODULE_STATE_LIVE = 0,
    MODULE_STATE_COMING = 1,
    MODULE_STATE_GOING = 2,
    MODULE_STATE_UNFORMED = 3
} ;
struct module_ref {
   unsigned long incs ;
   unsigned long decs ;
};
struct module_sect_attrs;
struct module_notes_attrs;
struct tracepoint;
struct ftrace_event_call;
struct module {
   enum module_state state ;
   struct list_head list ;
   char name[56U] ;
   struct module_kobject mkobj ;
   struct module_attribute *modinfo_attrs ;
   char const *version ;
   char const *srcversion ;
   struct kobject *holders_dir ;
   struct kernel_symbol const *syms ;
   unsigned long const *crcs ;
   unsigned int num_syms ;
   struct kernel_param *kp ;
   unsigned int num_kp ;
   unsigned int num_gpl_syms ;
   struct kernel_symbol const *gpl_syms ;
   unsigned long const *gpl_crcs ;
   struct kernel_symbol const *unused_syms ;
   unsigned long const *unused_crcs ;
   unsigned int num_unused_syms ;
   unsigned int num_unused_gpl_syms ;
   struct kernel_symbol const *unused_gpl_syms ;
   unsigned long const *unused_gpl_crcs ;
   bool sig_ok ;
   struct kernel_symbol const *gpl_future_syms ;
   unsigned long const *gpl_future_crcs ;
   unsigned int num_gpl_future_syms ;
   unsigned int num_exentries ;
   struct exception_table_entry *extable ;
   int (*init)(void) ;
   void *module_init ;
   void *module_core ;
   unsigned int init_size ;
   unsigned int core_size ;
   unsigned int init_text_size ;
   unsigned int core_text_size ;
   unsigned int init_ro_size ;
   unsigned int core_ro_size ;
   struct mod_arch_specific arch ;
   unsigned int taints ;
   unsigned int num_bugs ;
   struct list_head bug_list ;
   struct bug_entry *bug_table ;
   Elf64_Sym *symtab ;
   Elf64_Sym *core_symtab ;
   unsigned int num_symtab ;
   unsigned int core_num_syms ;
   char *strtab ;
   char *core_strtab ;
   struct module_sect_attrs *sect_attrs ;
   struct module_notes_attrs *notes_attrs ;
   char *args ;
   void *percpu ;
   unsigned int percpu_size ;
   unsigned int num_tracepoints ;
   struct tracepoint * const *tracepoints_ptrs ;
   unsigned int num_trace_bprintk_fmt ;
   char const **trace_bprintk_fmt_start ;
   struct ftrace_event_call **trace_events ;
   unsigned int num_trace_events ;
   unsigned int num_ftrace_callsites ;
   unsigned long *ftrace_callsites ;
   struct list_head source_list ;
   struct list_head target_list ;
   void (*exit)(void) ;
   struct module_ref *refptr ;
   ctor_fn_t (**ctors)(void) ;
   unsigned int num_ctors ;
};
struct mem_cgroup;
struct kmem_cache_cpu {
   void **freelist ;
   unsigned long tid ;
   struct page *page ;
   struct page *partial ;
   unsigned int stat[26U] ;
};
struct kmem_cache_order_objects {
   unsigned long x ;
};
struct memcg_cache_params;
struct kmem_cache_node;
struct kmem_cache {
   struct kmem_cache_cpu *cpu_slab ;
   unsigned long flags ;
   unsigned long min_partial ;
   int size ;
   int object_size ;
   int offset ;
   int cpu_partial ;
   struct kmem_cache_order_objects oo ;
   struct kmem_cache_order_objects max ;
   struct kmem_cache_order_objects min ;
   gfp_t allocflags ;
   int refcount ;
   void (*ctor)(void * ) ;
   int inuse ;
   int align ;
   int reserved ;
   char const *name ;
   struct list_head list ;
   struct kobject kobj ;
   struct memcg_cache_params *memcg_params ;
   int max_attr_size ;
   struct kset *memcg_kset ;
   int remote_node_defrag_ratio ;
   struct kmem_cache_node *node[1024U] ;
};
struct __anonstruct_ldv_15963_157 {
   struct callback_head callback_head ;
   struct kmem_cache *memcg_caches[0U] ;
};
struct __anonstruct_ldv_15969_158 {
   struct mem_cgroup *memcg ;
   struct list_head list ;
   struct kmem_cache *root_cache ;
   atomic_t nr_pages ;
};
union __anonunion_ldv_15970_156 {
   struct __anonstruct_ldv_15963_157 ldv_15963 ;
   struct __anonstruct_ldv_15969_158 ldv_15969 ;
};
struct memcg_cache_params {
   bool is_root_cache ;
   union __anonunion_ldv_15970_156 ldv_15970 ;
};
enum irqreturn {
    IRQ_NONE = 0,
    IRQ_HANDLED = 1,
    IRQ_WAKE_THREAD = 2
} ;
typedef enum irqreturn irqreturn_t;
struct urb;
struct usb_hcd;
struct spi_device;
struct klist_node;
struct klist_node {
   void *n_klist ;
   struct list_head n_node ;
   struct kref n_ref ;
};
struct path;
struct seq_file {
   char *buf ;
   size_t size ;
   size_t from ;
   size_t count ;
   size_t pad_until ;
   loff_t index ;
   loff_t read_pos ;
   u64 version ;
   struct mutex lock ;
   struct seq_operations const *op ;
   int poll_event ;
   struct user_namespace *user_ns ;
   void *private ;
};
struct seq_operations {
   void *(*start)(struct seq_file * , loff_t * ) ;
   void (*stop)(struct seq_file * , void * ) ;
   void *(*next)(struct seq_file * , void * , loff_t * ) ;
   int (*show)(struct seq_file * , void * ) ;
};
struct pinctrl;
struct pinctrl_state;
struct dev_pin_info {
   struct pinctrl *p ;
   struct pinctrl_state *default_state ;
   struct pinctrl_state *sleep_state ;
   struct pinctrl_state *idle_state ;
};
struct dma_map_ops;
struct dev_archdata {
   struct dma_map_ops *dma_ops ;
   void *iommu ;
};
struct device_private;
struct device_driver;
struct driver_private;
struct class;
struct subsys_private;
struct bus_type;
struct device_node;
struct iommu_ops;
struct iommu_group;
struct device_attribute;
struct bus_type {
   char const *name ;
   char const *dev_name ;
   struct device *dev_root ;
   struct device_attribute *dev_attrs ;
   struct attribute_group const **bus_groups ;
   struct attribute_group const **dev_groups ;
   struct attribute_group const **drv_groups ;
   int (*match)(struct device * , struct device_driver * ) ;
   int (*uevent)(struct device * , struct kobj_uevent_env * ) ;
   int (*probe)(struct device * ) ;
   int (*remove)(struct device * ) ;
   void (*shutdown)(struct device * ) ;
   int (*online)(struct device * ) ;
   int (*offline)(struct device * ) ;
   int (*suspend)(struct device * , pm_message_t ) ;
   int (*resume)(struct device * ) ;
   struct dev_pm_ops const *pm ;
   struct iommu_ops *iommu_ops ;
   struct subsys_private *p ;
   struct lock_class_key lock_key ;
};
struct device_type;
struct of_device_id;
struct acpi_device_id;
struct device_driver {
   char const *name ;
   struct bus_type *bus ;
   struct module *owner ;
   char const *mod_name ;
   bool suppress_bind_attrs ;
   struct of_device_id const *of_match_table ;
   struct acpi_device_id const *acpi_match_table ;
   int (*probe)(struct device * ) ;
   int (*remove)(struct device * ) ;
   void (*shutdown)(struct device * ) ;
   int (*suspend)(struct device * , pm_message_t ) ;
   int (*resume)(struct device * ) ;
   struct attribute_group const **groups ;
   struct dev_pm_ops const *pm ;
   struct driver_private *p ;
};
struct class_attribute;
struct class {
   char const *name ;
   struct module *owner ;
   struct class_attribute *class_attrs ;
   struct attribute_group const **dev_groups ;
   struct kobject *dev_kobj ;
   int (*dev_uevent)(struct device * , struct kobj_uevent_env * ) ;
   char *(*devnode)(struct device * , umode_t * ) ;
   void (*class_release)(struct class * ) ;
   void (*dev_release)(struct device * ) ;
   int (*suspend)(struct device * , pm_message_t ) ;
   int (*resume)(struct device * ) ;
   struct kobj_ns_type_operations const *ns_type ;
   void const *(*namespace)(struct device * ) ;
   struct dev_pm_ops const *pm ;
   struct subsys_private *p ;
};
struct class_attribute {
   struct attribute attr ;
   ssize_t (*show)(struct class * , struct class_attribute * , char * ) ;
   ssize_t (*store)(struct class * , struct class_attribute * , char const * , size_t ) ;
};
struct device_type {
   char const *name ;
   struct attribute_group const **groups ;
   int (*uevent)(struct device * , struct kobj_uevent_env * ) ;
   char *(*devnode)(struct device * , umode_t * , kuid_t * , kgid_t * ) ;
   void (*release)(struct device * ) ;
   struct dev_pm_ops const *pm ;
};
struct device_attribute {
   struct attribute attr ;
   ssize_t (*show)(struct device * , struct device_attribute * , char * ) ;
   ssize_t (*store)(struct device * , struct device_attribute * , char const * ,
                    size_t ) ;
};
struct device_dma_parameters {
   unsigned int max_segment_size ;
   unsigned long segment_boundary_mask ;
};
struct acpi_device;
struct acpi_dev_node {
   struct acpi_device *companion ;
};
struct dma_coherent_mem;
struct cma;
struct device {
   struct device *parent ;
   struct device_private *p ;
   struct kobject kobj ;
   char const *init_name ;
   struct device_type const *type ;
   struct mutex mutex ;
   struct bus_type *bus ;
   struct device_driver *driver ;
   void *platform_data ;
   void *driver_data ;
   struct dev_pm_info power ;
   struct dev_pm_domain *pm_domain ;
   struct dev_pin_info *pins ;
   int numa_node ;
   u64 *dma_mask ;
   u64 coherent_dma_mask ;
   unsigned long dma_pfn_offset ;
   struct device_dma_parameters *dma_parms ;
   struct list_head dma_pools ;
   struct dma_coherent_mem *dma_mem ;
   struct cma *cma_area ;
   struct dev_archdata archdata ;
   struct device_node *of_node ;
   struct acpi_dev_node acpi_node ;
   dev_t devt ;
   u32 id ;
   spinlock_t devres_lock ;
   struct list_head devres_head ;
   struct klist_node knode_class ;
   struct class *class ;
   struct attribute_group const **groups ;
   void (*release)(struct device * ) ;
   struct iommu_group *iommu_group ;
   bool offline_disabled ;
   bool offline ;
};
struct wakeup_source {
   char const *name ;
   struct list_head entry ;
   spinlock_t lock ;
   struct timer_list timer ;
   unsigned long timer_expires ;
   ktime_t total_time ;
   ktime_t max_time ;
   ktime_t last_time ;
   ktime_t start_prevent_time ;
   ktime_t prevent_sleep_time ;
   unsigned long event_count ;
   unsigned long active_count ;
   unsigned long relax_count ;
   unsigned long expire_count ;
   unsigned long wakeup_count ;
   bool active ;
   bool autosleep_enabled ;
};
typedef unsigned long kernel_ulong_t;
struct acpi_device_id {
   __u8 id[9U] ;
   kernel_ulong_t driver_data ;
};
struct of_device_id {
   char name[32U] ;
   char type[32U] ;
   char compatible[128U] ;
   void const *data ;
};
struct spi_device_id {
   char name[32U] ;
   kernel_ulong_t driver_data ;
};
struct kernel_cap_struct {
   __u32 cap[2U] ;
};
typedef struct kernel_cap_struct kernel_cap_t;
struct plist_node {
   int prio ;
   struct list_head prio_list ;
   struct list_head node_list ;
};
typedef unsigned long cputime_t;
struct sem_undo_list;
struct sysv_sem {
   struct sem_undo_list *undo_list ;
};
struct __anonstruct_sigset_t_163 {
   unsigned long sig[1U] ;
};
typedef struct __anonstruct_sigset_t_163 sigset_t;
struct siginfo;
typedef void __signalfn_t(int );
typedef __signalfn_t *__sighandler_t;
typedef void __restorefn_t(void);
typedef __restorefn_t *__sigrestore_t;
union sigval {
   int sival_int ;
   void *sival_ptr ;
};
typedef union sigval sigval_t;
struct __anonstruct__kill_165 {
   __kernel_pid_t _pid ;
   __kernel_uid32_t _uid ;
};
struct __anonstruct__timer_166 {
   __kernel_timer_t _tid ;
   int _overrun ;
   char _pad[0U] ;
   sigval_t _sigval ;
   int _sys_private ;
};
struct __anonstruct__rt_167 {
   __kernel_pid_t _pid ;
   __kernel_uid32_t _uid ;
   sigval_t _sigval ;
};
struct __anonstruct__sigchld_168 {
   __kernel_pid_t _pid ;
   __kernel_uid32_t _uid ;
   int _status ;
   __kernel_clock_t _utime ;
   __kernel_clock_t _stime ;
};
struct __anonstruct__sigfault_169 {
   void *_addr ;
   short _addr_lsb ;
};
struct __anonstruct__sigpoll_170 {
   long _band ;
   int _fd ;
};
struct __anonstruct__sigsys_171 {
   void *_call_addr ;
   int _syscall ;
   unsigned int _arch ;
};
union __anonunion__sifields_164 {
   int _pad[28U] ;
   struct __anonstruct__kill_165 _kill ;
   struct __anonstruct__timer_166 _timer ;
   struct __anonstruct__rt_167 _rt ;
   struct __anonstruct__sigchld_168 _sigchld ;
   struct __anonstruct__sigfault_169 _sigfault ;
   struct __anonstruct__sigpoll_170 _sigpoll ;
   struct __anonstruct__sigsys_171 _sigsys ;
};
struct siginfo {
   int si_signo ;
   int si_errno ;
   int si_code ;
   union __anonunion__sifields_164 _sifields ;
};
typedef struct siginfo siginfo_t;
struct user_struct;
struct sigpending {
   struct list_head list ;
   sigset_t signal ;
};
struct sigaction {
   __sighandler_t sa_handler ;
   unsigned long sa_flags ;
   __sigrestore_t sa_restorer ;
   sigset_t sa_mask ;
};
struct k_sigaction {
   struct sigaction sa ;
};
enum pid_type {
    PIDTYPE_PID = 0,
    PIDTYPE_PGID = 1,
    PIDTYPE_SID = 2,
    PIDTYPE_MAX = 3
} ;
struct pid_namespace;
struct upid {
   int nr ;
   struct pid_namespace *ns ;
   struct hlist_node pid_chain ;
};
struct pid {
   atomic_t count ;
   unsigned int level ;
   struct hlist_head tasks[3U] ;
   struct callback_head rcu ;
   struct upid numbers[1U] ;
};
struct pid_link {
   struct hlist_node node ;
   struct pid *pid ;
};
struct percpu_counter {
   raw_spinlock_t lock ;
   s64 count ;
   struct list_head list ;
   s32 *counters ;
};
struct seccomp_filter;
struct seccomp {
   int mode ;
   struct seccomp_filter *filter ;
};
struct rt_mutex_waiter;
struct rlimit {
   __kernel_ulong_t rlim_cur ;
   __kernel_ulong_t rlim_max ;
};
struct timerqueue_node {
   struct rb_node node ;
   ktime_t expires ;
};
struct timerqueue_head {
   struct rb_root head ;
   struct timerqueue_node *next ;
};
struct hrtimer_clock_base;
struct hrtimer_cpu_base;
enum hrtimer_restart {
    HRTIMER_NORESTART = 0,
    HRTIMER_RESTART = 1
} ;
struct hrtimer {
   struct timerqueue_node node ;
   ktime_t _softexpires ;
   enum hrtimer_restart (*function)(struct hrtimer * ) ;
   struct hrtimer_clock_base *base ;
   unsigned long state ;
   int start_pid ;
   void *start_site ;
   char start_comm[16U] ;
};
struct hrtimer_clock_base {
   struct hrtimer_cpu_base *cpu_base ;
   int index ;
   clockid_t clockid ;
   struct timerqueue_head active ;
   ktime_t resolution ;
   ktime_t (*get_time)(void) ;
   ktime_t softirq_time ;
   ktime_t offset ;
};
struct hrtimer_cpu_base {
   raw_spinlock_t lock ;
   unsigned int active_bases ;
   unsigned int clock_was_set ;
   ktime_t expires_next ;
   int hres_active ;
   int hang_detected ;
   unsigned long nr_events ;
   unsigned long nr_retries ;
   unsigned long nr_hangs ;
   ktime_t max_hang_time ;
   struct hrtimer_clock_base clock_base[4U] ;
};
struct task_io_accounting {
   u64 rchar ;
   u64 wchar ;
   u64 syscr ;
   u64 syscw ;
   u64 read_bytes ;
   u64 write_bytes ;
   u64 cancelled_write_bytes ;
};
struct latency_record {
   unsigned long backtrace[12U] ;
   unsigned int count ;
   unsigned long time ;
   unsigned long max ;
};
struct assoc_array_ptr;
struct assoc_array {
   struct assoc_array_ptr *root ;
   unsigned long nr_leaves_on_tree ;
};
typedef int32_t key_serial_t;
typedef uint32_t key_perm_t;
struct key;
struct signal_struct;
struct key_type;
struct keyring_index_key {
   struct key_type *type ;
   char const *description ;
   size_t desc_len ;
};
union __anonunion_ldv_18983_174 {
   struct list_head graveyard_link ;
   struct rb_node serial_node ;
};
struct key_user;
union __anonunion_ldv_18991_175 {
   time_t expiry ;
   time_t revoked_at ;
};
struct __anonstruct_ldv_19004_177 {
   struct key_type *type ;
   char *description ;
};
union __anonunion_ldv_19005_176 {
   struct keyring_index_key index_key ;
   struct __anonstruct_ldv_19004_177 ldv_19004 ;
};
union __anonunion_type_data_178 {
   struct list_head link ;
   unsigned long x[2U] ;
   void *p[2U] ;
   int reject_error ;
};
union __anonunion_payload_180 {
   unsigned long value ;
   void *rcudata ;
   void *data ;
   void *data2[2U] ;
};
union __anonunion_ldv_19020_179 {
   union __anonunion_payload_180 payload ;
   struct assoc_array keys ;
};
struct key {
   atomic_t usage ;
   key_serial_t serial ;
   union __anonunion_ldv_18983_174 ldv_18983 ;
   struct rw_semaphore sem ;
   struct key_user *user ;
   void *security ;
   union __anonunion_ldv_18991_175 ldv_18991 ;
   time_t last_used_at ;
   kuid_t uid ;
   kgid_t gid ;
   key_perm_t perm ;
   unsigned short quotalen ;
   unsigned short datalen ;
   unsigned long flags ;
   union __anonunion_ldv_19005_176 ldv_19005 ;
   union __anonunion_type_data_178 type_data ;
   union __anonunion_ldv_19020_179 ldv_19020 ;
};
struct audit_context;
struct group_info {
   atomic_t usage ;
   int ngroups ;
   int nblocks ;
   kgid_t small_block[32U] ;
   kgid_t *blocks[0U] ;
};
struct cred {
   atomic_t usage ;
   atomic_t subscribers ;
   void *put_addr ;
   unsigned int magic ;
   kuid_t uid ;
   kgid_t gid ;
   kuid_t suid ;
   kgid_t sgid ;
   kuid_t euid ;
   kgid_t egid ;
   kuid_t fsuid ;
   kgid_t fsgid ;
   unsigned int securebits ;
   kernel_cap_t cap_inheritable ;
   kernel_cap_t cap_permitted ;
   kernel_cap_t cap_effective ;
   kernel_cap_t cap_bset ;
   unsigned char jit_keyring ;
   struct key *session_keyring ;
   struct key *process_keyring ;
   struct key *thread_keyring ;
   struct key *request_key_auth ;
   void *security ;
   struct user_struct *user ;
   struct user_namespace *user_ns ;
   struct group_info *group_info ;
   struct callback_head rcu ;
};
struct futex_pi_state;
struct robust_list_head;
struct bio_list;
struct fs_struct;
struct perf_event_context;
struct blk_plug;
struct cfs_rq;
struct task_group;
struct sighand_struct {
   atomic_t count ;
   struct k_sigaction action[64U] ;
   spinlock_t siglock ;
   wait_queue_head_t signalfd_wqh ;
};
struct pacct_struct {
   int ac_flag ;
   long ac_exitcode ;
   unsigned long ac_mem ;
   cputime_t ac_utime ;
   cputime_t ac_stime ;
   unsigned long ac_minflt ;
   unsigned long ac_majflt ;
};
struct cpu_itimer {
   cputime_t expires ;
   cputime_t incr ;
   u32 error ;
   u32 incr_error ;
};
struct cputime {
   cputime_t utime ;
   cputime_t stime ;
};
struct task_cputime {
   cputime_t utime ;
   cputime_t stime ;
   unsigned long long sum_exec_runtime ;
};
struct thread_group_cputimer {
   struct task_cputime cputime ;
   int running ;
   raw_spinlock_t lock ;
};
struct autogroup;
struct tty_struct;
struct taskstats;
struct tty_audit_buf;
struct signal_struct {
   atomic_t sigcnt ;
   atomic_t live ;
   int nr_threads ;
   struct list_head thread_head ;
   wait_queue_head_t wait_chldexit ;
   struct task_struct *curr_target ;
   struct sigpending shared_pending ;
   int group_exit_code ;
   int notify_count ;
   struct task_struct *group_exit_task ;
   int group_stop_count ;
   unsigned int flags ;
   unsigned char is_child_subreaper : 1 ;
   unsigned char has_child_subreaper : 1 ;
   int posix_timer_id ;
   struct list_head posix_timers ;
   struct hrtimer real_timer ;
   struct pid *leader_pid ;
   ktime_t it_real_incr ;
   struct cpu_itimer it[2U] ;
   struct thread_group_cputimer cputimer ;
   struct task_cputime cputime_expires ;
   struct list_head cpu_timers[3U] ;
   struct pid *tty_old_pgrp ;
   int leader ;
   struct tty_struct *tty ;
   struct autogroup *autogroup ;
   cputime_t utime ;
   cputime_t stime ;
   cputime_t cutime ;
   cputime_t cstime ;
   cputime_t gtime ;
   cputime_t cgtime ;
   struct cputime prev_cputime ;
   unsigned long nvcsw ;
   unsigned long nivcsw ;
   unsigned long cnvcsw ;
   unsigned long cnivcsw ;
   unsigned long min_flt ;
   unsigned long maj_flt ;
   unsigned long cmin_flt ;
   unsigned long cmaj_flt ;
   unsigned long inblock ;
   unsigned long oublock ;
   unsigned long cinblock ;
   unsigned long coublock ;
   unsigned long maxrss ;
   unsigned long cmaxrss ;
   struct task_io_accounting ioac ;
   unsigned long long sum_sched_runtime ;
   struct rlimit rlim[16U] ;
   struct pacct_struct pacct ;
   struct taskstats *stats ;
   unsigned int audit_tty ;
   unsigned int audit_tty_log_passwd ;
   struct tty_audit_buf *tty_audit_buf ;
   struct rw_semaphore group_rwsem ;
   oom_flags_t oom_flags ;
   short oom_score_adj ;
   short oom_score_adj_min ;
   struct mutex cred_guard_mutex ;
};
struct user_struct {
   atomic_t __count ;
   atomic_t processes ;
   atomic_t sigpending ;
   atomic_t inotify_watches ;
   atomic_t inotify_devs ;
   atomic_t fanotify_listeners ;
   atomic_long_t epoll_watches ;
   unsigned long mq_bytes ;
   unsigned long locked_shm ;
   struct key *uid_keyring ;
   struct key *session_keyring ;
   struct hlist_node uidhash_node ;
   kuid_t uid ;
   atomic_long_t locked_vm ;
};
struct backing_dev_info;
struct reclaim_state;
struct sched_info {
   unsigned long pcount ;
   unsigned long long run_delay ;
   unsigned long long last_arrival ;
   unsigned long long last_queued ;
};
struct task_delay_info {
   spinlock_t lock ;
   unsigned int flags ;
   struct timespec blkio_start ;
   struct timespec blkio_end ;
   u64 blkio_delay ;
   u64 swapin_delay ;
   u32 blkio_count ;
   u32 swapin_count ;
   struct timespec freepages_start ;
   struct timespec freepages_end ;
   u64 freepages_delay ;
   u32 freepages_count ;
};
struct io_context;
struct pipe_inode_info;
struct load_weight {
   unsigned long weight ;
   u32 inv_weight ;
};
struct sched_avg {
   u32 runnable_avg_sum ;
   u32 runnable_avg_period ;
   u64 last_runnable_update ;
   s64 decay_count ;
   unsigned long load_avg_contrib ;
};
struct sched_statistics {
   u64 wait_start ;
   u64 wait_max ;
   u64 wait_count ;
   u64 wait_sum ;
   u64 iowait_count ;
   u64 iowait_sum ;
   u64 sleep_start ;
   u64 sleep_max ;
   s64 sum_sleep_runtime ;
   u64 block_start ;
   u64 block_max ;
   u64 exec_max ;
   u64 slice_max ;
   u64 nr_migrations_cold ;
   u64 nr_failed_migrations_affine ;
   u64 nr_failed_migrations_running ;
   u64 nr_failed_migrations_hot ;
   u64 nr_forced_migrations ;
   u64 nr_wakeups ;
   u64 nr_wakeups_sync ;
   u64 nr_wakeups_migrate ;
   u64 nr_wakeups_local ;
   u64 nr_wakeups_remote ;
   u64 nr_wakeups_affine ;
   u64 nr_wakeups_affine_attempts ;
   u64 nr_wakeups_passive ;
   u64 nr_wakeups_idle ;
};
struct sched_entity {
   struct load_weight load ;
   struct rb_node run_node ;
   struct list_head group_node ;
   unsigned int on_rq ;
   u64 exec_start ;
   u64 sum_exec_runtime ;
   u64 vruntime ;
   u64 prev_sum_exec_runtime ;
   u64 nr_migrations ;
   struct sched_statistics statistics ;
   int depth ;
   struct sched_entity *parent ;
   struct cfs_rq *cfs_rq ;
   struct cfs_rq *my_q ;
   struct sched_avg avg ;
};
struct rt_rq;
struct sched_rt_entity {
   struct list_head run_list ;
   unsigned long timeout ;
   unsigned long watchdog_stamp ;
   unsigned int time_slice ;
   struct sched_rt_entity *back ;
   struct sched_rt_entity *parent ;
   struct rt_rq *rt_rq ;
   struct rt_rq *my_q ;
};
struct sched_dl_entity {
   struct rb_node rb_node ;
   u64 dl_runtime ;
   u64 dl_deadline ;
   u64 dl_period ;
   u64 dl_bw ;
   s64 runtime ;
   u64 deadline ;
   unsigned int flags ;
   int dl_throttled ;
   int dl_new ;
   int dl_boosted ;
   int dl_yielded ;
   struct hrtimer dl_timer ;
};
struct memcg_batch_info {
   int do_batch ;
   struct mem_cgroup *memcg ;
   unsigned long nr_pages ;
   unsigned long memsw_nr_pages ;
};
struct memcg_oom_info {
   struct mem_cgroup *memcg ;
   gfp_t gfp_mask ;
   int order ;
   unsigned char may_oom : 1 ;
};
struct sched_class;
struct files_struct;
struct css_set;
struct compat_robust_list_head;
struct numa_group;
struct ftrace_ret_stack;
struct task_struct {
   long volatile state ;
   void *stack ;
   atomic_t usage ;
   unsigned int flags ;
   unsigned int ptrace ;
   struct llist_node wake_entry ;
   int on_cpu ;
   struct task_struct *last_wakee ;
   unsigned long wakee_flips ;
   unsigned long wakee_flip_decay_ts ;
   int wake_cpu ;
   int on_rq ;
   int prio ;
   int static_prio ;
   int normal_prio ;
   unsigned int rt_priority ;
   struct sched_class const *sched_class ;
   struct sched_entity se ;
   struct sched_rt_entity rt ;
   struct task_group *sched_task_group ;
   struct sched_dl_entity dl ;
   struct hlist_head preempt_notifiers ;
   unsigned int btrace_seq ;
   unsigned int policy ;
   int nr_cpus_allowed ;
   cpumask_t cpus_allowed ;
   struct sched_info sched_info ;
   struct list_head tasks ;
   struct plist_node pushable_tasks ;
   struct rb_node pushable_dl_tasks ;
   struct mm_struct *mm ;
   struct mm_struct *active_mm ;
   unsigned char brk_randomized : 1 ;
   u32 vmacache_seqnum ;
   struct vm_area_struct *vmacache[4U] ;
   struct task_rss_stat rss_stat ;
   int exit_state ;
   int exit_code ;
   int exit_signal ;
   int pdeath_signal ;
   unsigned int jobctl ;
   unsigned int personality ;
   unsigned char in_execve : 1 ;
   unsigned char in_iowait : 1 ;
   unsigned char no_new_privs : 1 ;
   unsigned char sched_reset_on_fork : 1 ;
   unsigned char sched_contributes_to_load : 1 ;
   pid_t pid ;
   pid_t tgid ;
   struct task_struct *real_parent ;
   struct task_struct *parent ;
   struct list_head children ;
   struct list_head sibling ;
   struct task_struct *group_leader ;
   struct list_head ptraced ;
   struct list_head ptrace_entry ;
   struct pid_link pids[3U] ;
   struct list_head thread_group ;
   struct list_head thread_node ;
   struct completion *vfork_done ;
   int *set_child_tid ;
   int *clear_child_tid ;
   cputime_t utime ;
   cputime_t stime ;
   cputime_t utimescaled ;
   cputime_t stimescaled ;
   cputime_t gtime ;
   struct cputime prev_cputime ;
   unsigned long nvcsw ;
   unsigned long nivcsw ;
   struct timespec start_time ;
   struct timespec real_start_time ;
   unsigned long min_flt ;
   unsigned long maj_flt ;
   struct task_cputime cputime_expires ;
   struct list_head cpu_timers[3U] ;
   struct cred const *real_cred ;
   struct cred const *cred ;
   char comm[16U] ;
   int link_count ;
   int total_link_count ;
   struct sysv_sem sysvsem ;
   unsigned long last_switch_count ;
   struct thread_struct thread ;
   struct fs_struct *fs ;
   struct files_struct *files ;
   struct nsproxy *nsproxy ;
   struct signal_struct *signal ;
   struct sighand_struct *sighand ;
   sigset_t blocked ;
   sigset_t real_blocked ;
   sigset_t saved_sigmask ;
   struct sigpending pending ;
   unsigned long sas_ss_sp ;
   size_t sas_ss_size ;
   int (*notifier)(void * ) ;
   void *notifier_data ;
   sigset_t *notifier_mask ;
   struct callback_head *task_works ;
   struct audit_context *audit_context ;
   kuid_t loginuid ;
   unsigned int sessionid ;
   struct seccomp seccomp ;
   u32 parent_exec_id ;
   u32 self_exec_id ;
   spinlock_t alloc_lock ;
   raw_spinlock_t pi_lock ;
   struct rb_root pi_waiters ;
   struct rb_node *pi_waiters_leftmost ;
   struct rt_mutex_waiter *pi_blocked_on ;
   struct task_struct *pi_top_task ;
   struct mutex_waiter *blocked_on ;
   unsigned int irq_events ;
   unsigned long hardirq_enable_ip ;
   unsigned long hardirq_disable_ip ;
   unsigned int hardirq_enable_event ;
   unsigned int hardirq_disable_event ;
   int hardirqs_enabled ;
   int hardirq_context ;
   unsigned long softirq_disable_ip ;
   unsigned long softirq_enable_ip ;
   unsigned int softirq_disable_event ;
   unsigned int softirq_enable_event ;
   int softirqs_enabled ;
   int softirq_context ;
   u64 curr_chain_key ;
   int lockdep_depth ;
   unsigned int lockdep_recursion ;
   struct held_lock held_locks[48U] ;
   gfp_t lockdep_reclaim_gfp ;
   void *journal_info ;
   struct bio_list *bio_list ;
   struct blk_plug *plug ;
   struct reclaim_state *reclaim_state ;
   struct backing_dev_info *backing_dev_info ;
   struct io_context *io_context ;
   unsigned long ptrace_message ;
   siginfo_t *last_siginfo ;
   struct task_io_accounting ioac ;
   u64 acct_rss_mem1 ;
   u64 acct_vm_mem1 ;
   cputime_t acct_timexpd ;
   nodemask_t mems_allowed ;
   seqcount_t mems_allowed_seq ;
   int cpuset_mem_spread_rotor ;
   int cpuset_slab_spread_rotor ;
   struct css_set *cgroups ;
   struct list_head cg_list ;
   struct robust_list_head *robust_list ;
   struct compat_robust_list_head *compat_robust_list ;
   struct list_head pi_state_list ;
   struct futex_pi_state *pi_state_cache ;
   struct perf_event_context *perf_event_ctxp[2U] ;
   struct mutex perf_event_mutex ;
   struct list_head perf_event_list ;
   struct mempolicy *mempolicy ;
   short il_next ;
   short pref_node_fork ;
   int numa_scan_seq ;
   unsigned int numa_scan_period ;
   unsigned int numa_scan_period_max ;
   int numa_preferred_nid ;
   unsigned long numa_migrate_retry ;
   u64 node_stamp ;
   u64 last_task_numa_placement ;
   u64 last_sum_exec_runtime ;
   struct callback_head numa_work ;
   struct list_head numa_entry ;
   struct numa_group *numa_group ;
   unsigned long *numa_faults_memory ;
   unsigned long total_numa_faults ;
   unsigned long *numa_faults_buffer_memory ;
   unsigned long *numa_faults_cpu ;
   unsigned long *numa_faults_buffer_cpu ;
   unsigned long numa_faults_locality[2U] ;
   unsigned long numa_pages_migrated ;
   struct callback_head rcu ;
   struct pipe_inode_info *splice_pipe ;
   struct page_frag task_frag ;
   struct task_delay_info *delays ;
   int make_it_fail ;
   int nr_dirtied ;
   int nr_dirtied_pause ;
   unsigned long dirty_paused_when ;
   int latency_record_count ;
   struct latency_record latency_record[32U] ;
   unsigned long timer_slack_ns ;
   unsigned long default_timer_slack_ns ;
   int curr_ret_stack ;
   struct ftrace_ret_stack *ret_stack ;
   unsigned long long ftrace_timestamp ;
   atomic_t trace_overrun ;
   atomic_t tracing_graph_pause ;
   unsigned long trace ;
   unsigned long trace_recursion ;
   struct memcg_batch_info memcg_batch ;
   unsigned int memcg_kmem_skip_account ;
   struct memcg_oom_info memcg_oom ;
   struct uprobe_task *utask ;
   unsigned int sequential_io ;
   unsigned int sequential_io_avg ;
};
struct kthread_work;
struct kthread_worker {
   spinlock_t lock ;
   struct list_head work_list ;
   struct task_struct *task ;
   struct kthread_work *current_work ;
};
struct kthread_work {
   struct list_head node ;
   void (*func)(struct kthread_work * ) ;
   wait_queue_head_t done ;
   struct kthread_worker *worker ;
};
struct shrink_control {
   gfp_t gfp_mask ;
   unsigned long nr_to_scan ;
   nodemask_t nodes_to_scan ;
   int nid ;
};
struct shrinker {
   unsigned long (*count_objects)(struct shrinker * , struct shrink_control * ) ;
   unsigned long (*scan_objects)(struct shrinker * , struct shrink_control * ) ;
   int seeks ;
   long batch ;
   unsigned long flags ;
   struct list_head list ;
   atomic_long_t *nr_deferred ;
};
struct file_ra_state;
struct writeback_control;
struct vm_fault {
   unsigned int flags ;
   unsigned long pgoff ;
   void *virtual_address ;
   struct page *page ;
   unsigned long max_pgoff ;
   pte_t *pte ;
};
struct vm_operations_struct {
   void (*open)(struct vm_area_struct * ) ;
   void (*close)(struct vm_area_struct * ) ;
   int (*fault)(struct vm_area_struct * , struct vm_fault * ) ;
   void (*map_pages)(struct vm_area_struct * , struct vm_fault * ) ;
   int (*page_mkwrite)(struct vm_area_struct * , struct vm_fault * ) ;
   int (*access)(struct vm_area_struct * , unsigned long , void * , int , int ) ;
   char const *(*name)(struct vm_area_struct * ) ;
   int (*set_policy)(struct vm_area_struct * , struct mempolicy * ) ;
   struct mempolicy *(*get_policy)(struct vm_area_struct * , unsigned long ) ;
   int (*migrate)(struct vm_area_struct * , nodemask_t const * , nodemask_t const * ,
                  unsigned long ) ;
   int (*remap_pages)(struct vm_area_struct * , unsigned long , unsigned long ,
                      unsigned long ) ;
};
struct scatterlist {
   unsigned long sg_magic ;
   unsigned long page_link ;
   unsigned int offset ;
   unsigned int length ;
   dma_addr_t dma_address ;
   unsigned int dma_length ;
};
struct sg_table {
   struct scatterlist *sgl ;
   unsigned int nents ;
   unsigned int orig_nents ;
};
struct dma_chan;
struct spi_master;
struct spi_device {
   struct device dev ;
   struct spi_master *master ;
   u32 max_speed_hz ;
   u8 chip_select ;
   u8 bits_per_word ;
   u16 mode ;
   int irq ;
   void *controller_state ;
   void *controller_data ;
   char modalias[32U] ;
   int cs_gpio ;
};
struct spi_message;
struct spi_transfer;
struct spi_driver {
   struct spi_device_id const *id_table ;
   int (*probe)(struct spi_device * ) ;
   int (*remove)(struct spi_device * ) ;
   void (*shutdown)(struct spi_device * ) ;
   int (*suspend)(struct spi_device * , pm_message_t ) ;
   int (*resume)(struct spi_device * ) ;
   struct device_driver driver ;
};
struct spi_master {
   struct device dev ;
   struct list_head list ;
   s16 bus_num ;
   u16 num_chipselect ;
   u16 dma_alignment ;
   u16 mode_bits ;
   u32 bits_per_word_mask ;
   u32 min_speed_hz ;
   u32 max_speed_hz ;
   u16 flags ;
   spinlock_t bus_lock_spinlock ;
   struct mutex bus_lock_mutex ;
   bool bus_lock_flag ;
   int (*setup)(struct spi_device * ) ;
   int (*transfer)(struct spi_device * , struct spi_message * ) ;
   void (*cleanup)(struct spi_device * ) ;
   bool (*can_dma)(struct spi_master * , struct spi_device * , struct spi_transfer * ) ;
   bool queued ;
   struct kthread_worker kworker ;
   struct task_struct *kworker_task ;
   struct kthread_work pump_messages ;
   spinlock_t queue_lock ;
   struct list_head queue ;
   struct spi_message *cur_msg ;
   bool busy ;
   bool running ;
   bool rt ;
   bool auto_runtime_pm ;
   bool cur_msg_prepared ;
   bool cur_msg_mapped ;
   struct completion xfer_completion ;
   size_t max_dma_len ;
   int (*prepare_transfer_hardware)(struct spi_master * ) ;
   int (*transfer_one_message)(struct spi_master * , struct spi_message * ) ;
   int (*unprepare_transfer_hardware)(struct spi_master * ) ;
   int (*prepare_message)(struct spi_master * , struct spi_message * ) ;
   int (*unprepare_message)(struct spi_master * , struct spi_message * ) ;
   void (*set_cs)(struct spi_device * , bool ) ;
   int (*transfer_one)(struct spi_master * , struct spi_device * , struct spi_transfer * ) ;
   int *cs_gpios ;
   struct dma_chan *dma_tx ;
   struct dma_chan *dma_rx ;
   void *dummy_rx ;
   void *dummy_tx ;
};
struct spi_transfer {
   void const *tx_buf ;
   void *rx_buf ;
   unsigned int len ;
   dma_addr_t tx_dma ;
   dma_addr_t rx_dma ;
   struct sg_table tx_sg ;
   struct sg_table rx_sg ;
   unsigned char cs_change : 1 ;
   unsigned char tx_nbits : 3 ;
   unsigned char rx_nbits : 3 ;
   u8 bits_per_word ;
   u16 delay_usecs ;
   u32 speed_hz ;
   struct list_head transfer_list ;
};
struct spi_message {
   struct list_head transfers ;
   struct spi_device *spi ;
   unsigned char is_dma_mapped : 1 ;
   void (*complete)(void * ) ;
   void *context ;
   unsigned int frame_length ;
   unsigned int actual_length ;
   int status ;
   struct list_head queue ;
   void *state ;
};
struct usb_device_descriptor {
   __u8 bLength ;
   __u8 bDescriptorType ;
   __le16 bcdUSB ;
   __u8 bDeviceClass ;
   __u8 bDeviceSubClass ;
   __u8 bDeviceProtocol ;
   __u8 bMaxPacketSize0 ;
   __le16 idVendor ;
   __le16 idProduct ;
   __le16 bcdDevice ;
   __u8 iManufacturer ;
   __u8 iProduct ;
   __u8 iSerialNumber ;
   __u8 bNumConfigurations ;
};
struct usb_config_descriptor {
   __u8 bLength ;
   __u8 bDescriptorType ;
   __le16 wTotalLength ;
   __u8 bNumInterfaces ;
   __u8 bConfigurationValue ;
   __u8 iConfiguration ;
   __u8 bmAttributes ;
   __u8 bMaxPower ;
};
struct usb_interface_descriptor {
   __u8 bLength ;
   __u8 bDescriptorType ;
   __u8 bInterfaceNumber ;
   __u8 bAlternateSetting ;
   __u8 bNumEndpoints ;
   __u8 bInterfaceClass ;
   __u8 bInterfaceSubClass ;
   __u8 bInterfaceProtocol ;
   __u8 iInterface ;
};
struct usb_endpoint_descriptor {
   __u8 bLength ;
   __u8 bDescriptorType ;
   __u8 bEndpointAddress ;
   __u8 bmAttributes ;
   __le16 wMaxPacketSize ;
   __u8 bInterval ;
   __u8 bRefresh ;
   __u8 bSynchAddress ;
};
struct usb_ss_ep_comp_descriptor {
   __u8 bLength ;
   __u8 bDescriptorType ;
   __u8 bMaxBurst ;
   __u8 bmAttributes ;
   __le16 wBytesPerInterval ;
};
struct usb_interface_assoc_descriptor {
   __u8 bLength ;
   __u8 bDescriptorType ;
   __u8 bFirstInterface ;
   __u8 bInterfaceCount ;
   __u8 bFunctionClass ;
   __u8 bFunctionSubClass ;
   __u8 bFunctionProtocol ;
   __u8 iFunction ;
};
struct usb_bos_descriptor {
   __u8 bLength ;
   __u8 bDescriptorType ;
   __le16 wTotalLength ;
   __u8 bNumDeviceCaps ;
};
struct usb_ext_cap_descriptor {
   __u8 bLength ;
   __u8 bDescriptorType ;
   __u8 bDevCapabilityType ;
   __le32 bmAttributes ;
};
struct usb_ss_cap_descriptor {
   __u8 bLength ;
   __u8 bDescriptorType ;
   __u8 bDevCapabilityType ;
   __u8 bmAttributes ;
   __le16 wSpeedSupported ;
   __u8 bFunctionalitySupport ;
   __u8 bU1devExitLat ;
   __le16 bU2DevExitLat ;
};
struct usb_ss_container_id_descriptor {
   __u8 bLength ;
   __u8 bDescriptorType ;
   __u8 bDevCapabilityType ;
   __u8 bReserved ;
   __u8 ContainerID[16U] ;
};
enum usb_device_speed {
    USB_SPEED_UNKNOWN = 0,
    USB_SPEED_LOW = 1,
    USB_SPEED_FULL = 2,
    USB_SPEED_HIGH = 3,
    USB_SPEED_WIRELESS = 4,
    USB_SPEED_SUPER = 5
} ;
enum usb_device_state {
    USB_STATE_NOTATTACHED = 0,
    USB_STATE_ATTACHED = 1,
    USB_STATE_POWERED = 2,
    USB_STATE_RECONNECTING = 3,
    USB_STATE_UNAUTHENTICATED = 4,
    USB_STATE_DEFAULT = 5,
    USB_STATE_ADDRESS = 6,
    USB_STATE_CONFIGURED = 7,
    USB_STATE_SUSPENDED = 8
} ;
enum usb3_link_state {
    USB3_LPM_U0 = 0,
    USB3_LPM_U1 = 1,
    USB3_LPM_U2 = 2,
    USB3_LPM_U3 = 3
} ;
struct exception_table_entry {
   int insn ;
   int fixup ;
};
struct tasklet_struct {
   struct tasklet_struct *next ;
   unsigned long state ;
   atomic_t count ;
   void (*func)(unsigned long ) ;
   unsigned long data ;
};
struct hlist_bl_node;
struct hlist_bl_head {
   struct hlist_bl_node *first ;
};
struct hlist_bl_node {
   struct hlist_bl_node *next ;
   struct hlist_bl_node **pprev ;
};
struct __anonstruct_ldv_26135_185 {
   spinlock_t lock ;
   unsigned int count ;
};
union __anonunion_ldv_26136_184 {
   struct __anonstruct_ldv_26135_185 ldv_26135 ;
};
struct lockref {
   union __anonunion_ldv_26136_184 ldv_26136 ;
};
struct nameidata;
struct vfsmount;
struct __anonstruct_ldv_26159_187 {
   u32 hash ;
   u32 len ;
};
union __anonunion_ldv_26161_186 {
   struct __anonstruct_ldv_26159_187 ldv_26159 ;
   u64 hash_len ;
};
struct qstr {
   union __anonunion_ldv_26161_186 ldv_26161 ;
   unsigned char const *name ;
};
struct dentry_operations;
union __anonunion_d_u_188 {
   struct list_head d_child ;
   struct callback_head d_rcu ;
};
struct dentry {
   unsigned int d_flags ;
   seqcount_t d_seq ;
   struct hlist_bl_node d_hash ;
   struct dentry *d_parent ;
   struct qstr d_name ;
   struct inode *d_inode ;
   unsigned char d_iname[32U] ;
   struct lockref d_lockref ;
   struct dentry_operations const *d_op ;
   struct super_block *d_sb ;
   unsigned long d_time ;
   void *d_fsdata ;
   struct list_head d_lru ;
   union __anonunion_d_u_188 d_u ;
   struct list_head d_subdirs ;
   struct hlist_node d_alias ;
};
struct dentry_operations {
   int (*d_revalidate)(struct dentry * , unsigned int ) ;
   int (*d_weak_revalidate)(struct dentry * , unsigned int ) ;
   int (*d_hash)(struct dentry const * , struct qstr * ) ;
   int (*d_compare)(struct dentry const * , struct dentry const * , unsigned int ,
                    char const * , struct qstr const * ) ;
   int (*d_delete)(struct dentry const * ) ;
   void (*d_release)(struct dentry * ) ;
   void (*d_prune)(struct dentry * ) ;
   void (*d_iput)(struct dentry * , struct inode * ) ;
   char *(*d_dname)(struct dentry * , char * , int ) ;
   struct vfsmount *(*d_automount)(struct path * ) ;
   int (*d_manage)(struct dentry * , bool ) ;
};
struct path {
   struct vfsmount *mnt ;
   struct dentry *dentry ;
};
struct list_lru_node {
   spinlock_t lock ;
   struct list_head list ;
   long nr_items ;
};
struct list_lru {
   struct list_lru_node *node ;
   nodemask_t active_nodes ;
};
struct __anonstruct_ldv_26522_190 {
   struct radix_tree_node *parent ;
   void *private_data ;
};
union __anonunion_ldv_26524_189 {
   struct __anonstruct_ldv_26522_190 ldv_26522 ;
   struct callback_head callback_head ;
};
struct radix_tree_node {
   unsigned int path ;
   unsigned int count ;
   union __anonunion_ldv_26524_189 ldv_26524 ;
   struct list_head private_list ;
   void *slots[64U] ;
   unsigned long tags[3U][1U] ;
};
struct radix_tree_root {
   unsigned int height ;
   gfp_t gfp_mask ;
   struct radix_tree_node *rnode ;
};
struct fiemap_extent {
   __u64 fe_logical ;
   __u64 fe_physical ;
   __u64 fe_length ;
   __u64 fe_reserved64[2U] ;
   __u32 fe_flags ;
   __u32 fe_reserved[3U] ;
};
enum migrate_mode {
    MIGRATE_ASYNC = 0,
    MIGRATE_SYNC_LIGHT = 1,
    MIGRATE_SYNC = 2
} ;
struct block_device;
struct export_operations;
struct iovec;
struct kiocb;
struct poll_table_struct;
struct kstatfs;
struct swap_info_struct;
struct iov_iter;
struct iattr {
   unsigned int ia_valid ;
   umode_t ia_mode ;
   kuid_t ia_uid ;
   kgid_t ia_gid ;
   loff_t ia_size ;
   struct timespec ia_atime ;
   struct timespec ia_mtime ;
   struct timespec ia_ctime ;
   struct file *ia_file ;
};
struct fs_disk_quota {
   __s8 d_version ;
   __s8 d_flags ;
   __u16 d_fieldmask ;
   __u32 d_id ;
   __u64 d_blk_hardlimit ;
   __u64 d_blk_softlimit ;
   __u64 d_ino_hardlimit ;
   __u64 d_ino_softlimit ;
   __u64 d_bcount ;
   __u64 d_icount ;
   __s32 d_itimer ;
   __s32 d_btimer ;
   __u16 d_iwarns ;
   __u16 d_bwarns ;
   __s32 d_padding2 ;
   __u64 d_rtb_hardlimit ;
   __u64 d_rtb_softlimit ;
   __u64 d_rtbcount ;
   __s32 d_rtbtimer ;
   __u16 d_rtbwarns ;
   __s16 d_padding3 ;
   char d_padding4[8U] ;
};
struct fs_qfilestat {
   __u64 qfs_ino ;
   __u64 qfs_nblks ;
   __u32 qfs_nextents ;
};
typedef struct fs_qfilestat fs_qfilestat_t;
struct fs_quota_stat {
   __s8 qs_version ;
   __u16 qs_flags ;
   __s8 qs_pad ;
   fs_qfilestat_t qs_uquota ;
   fs_qfilestat_t qs_gquota ;
   __u32 qs_incoredqs ;
   __s32 qs_btimelimit ;
   __s32 qs_itimelimit ;
   __s32 qs_rtbtimelimit ;
   __u16 qs_bwarnlimit ;
   __u16 qs_iwarnlimit ;
};
struct fs_qfilestatv {
   __u64 qfs_ino ;
   __u64 qfs_nblks ;
   __u32 qfs_nextents ;
   __u32 qfs_pad ;
};
struct fs_quota_statv {
   __s8 qs_version ;
   __u8 qs_pad1 ;
   __u16 qs_flags ;
   __u32 qs_incoredqs ;
   struct fs_qfilestatv qs_uquota ;
   struct fs_qfilestatv qs_gquota ;
   struct fs_qfilestatv qs_pquota ;
   __s32 qs_btimelimit ;
   __s32 qs_itimelimit ;
   __s32 qs_rtbtimelimit ;
   __u16 qs_bwarnlimit ;
   __u16 qs_iwarnlimit ;
   __u64 qs_pad2[8U] ;
};
struct dquot;
typedef __kernel_uid32_t projid_t;
struct __anonstruct_kprojid_t_191 {
   projid_t val ;
};
typedef struct __anonstruct_kprojid_t_191 kprojid_t;
struct if_dqinfo {
   __u64 dqi_bgrace ;
   __u64 dqi_igrace ;
   __u32 dqi_flags ;
   __u32 dqi_valid ;
};
enum quota_type {
    USRQUOTA = 0,
    GRPQUOTA = 1,
    PRJQUOTA = 2
} ;
typedef long long qsize_t;
union __anonunion_ldv_27053_192 {
   kuid_t uid ;
   kgid_t gid ;
   kprojid_t projid ;
};
struct kqid {
   union __anonunion_ldv_27053_192 ldv_27053 ;
   enum quota_type type ;
};
struct mem_dqblk {
   qsize_t dqb_bhardlimit ;
   qsize_t dqb_bsoftlimit ;
   qsize_t dqb_curspace ;
   qsize_t dqb_rsvspace ;
   qsize_t dqb_ihardlimit ;
   qsize_t dqb_isoftlimit ;
   qsize_t dqb_curinodes ;
   time_t dqb_btime ;
   time_t dqb_itime ;
};
struct quota_format_type;
struct mem_dqinfo {
   struct quota_format_type *dqi_format ;
   int dqi_fmt_id ;
   struct list_head dqi_dirty_list ;
   unsigned long dqi_flags ;
   unsigned int dqi_bgrace ;
   unsigned int dqi_igrace ;
   qsize_t dqi_maxblimit ;
   qsize_t dqi_maxilimit ;
   void *dqi_priv ;
};
struct dquot {
   struct hlist_node dq_hash ;
   struct list_head dq_inuse ;
   struct list_head dq_free ;
   struct list_head dq_dirty ;
   struct mutex dq_lock ;
   atomic_t dq_count ;
   wait_queue_head_t dq_wait_unused ;
   struct super_block *dq_sb ;
   struct kqid dq_id ;
   loff_t dq_off ;
   unsigned long dq_flags ;
   struct mem_dqblk dq_dqb ;
};
struct quota_format_ops {
   int (*check_quota_file)(struct super_block * , int ) ;
   int (*read_file_info)(struct super_block * , int ) ;
   int (*write_file_info)(struct super_block * , int ) ;
   int (*free_file_info)(struct super_block * , int ) ;
   int (*read_dqblk)(struct dquot * ) ;
   int (*commit_dqblk)(struct dquot * ) ;
   int (*release_dqblk)(struct dquot * ) ;
};
struct dquot_operations {
   int (*write_dquot)(struct dquot * ) ;
   struct dquot *(*alloc_dquot)(struct super_block * , int ) ;
   void (*destroy_dquot)(struct dquot * ) ;
   int (*acquire_dquot)(struct dquot * ) ;
   int (*release_dquot)(struct dquot * ) ;
   int (*mark_dirty)(struct dquot * ) ;
   int (*write_info)(struct super_block * , int ) ;
   qsize_t *(*get_reserved_space)(struct inode * ) ;
};
struct quotactl_ops {
   int (*quota_on)(struct super_block * , int , int , struct path * ) ;
   int (*quota_on_meta)(struct super_block * , int , int ) ;
   int (*quota_off)(struct super_block * , int ) ;
   int (*quota_sync)(struct super_block * , int ) ;
   int (*get_info)(struct super_block * , int , struct if_dqinfo * ) ;
   int (*set_info)(struct super_block * , int , struct if_dqinfo * ) ;
   int (*get_dqblk)(struct super_block * , struct kqid , struct fs_disk_quota * ) ;
   int (*set_dqblk)(struct super_block * , struct kqid , struct fs_disk_quota * ) ;
   int (*get_xstate)(struct super_block * , struct fs_quota_stat * ) ;
   int (*set_xstate)(struct super_block * , unsigned int , int ) ;
   int (*get_xstatev)(struct super_block * , struct fs_quota_statv * ) ;
   int (*rm_xquota)(struct super_block * , unsigned int ) ;
};
struct quota_format_type {
   int qf_fmt_id ;
   struct quota_format_ops const *qf_ops ;
   struct module *qf_owner ;
   struct quota_format_type *qf_next ;
};
struct quota_info {
   unsigned int flags ;
   struct mutex dqio_mutex ;
   struct mutex dqonoff_mutex ;
   struct rw_semaphore dqptr_sem ;
   struct inode *files[2U] ;
   struct mem_dqinfo info[2U] ;
   struct quota_format_ops const *ops[2U] ;
};
struct address_space_operations {
   int (*writepage)(struct page * , struct writeback_control * ) ;
   int (*readpage)(struct file * , struct page * ) ;
   int (*writepages)(struct address_space * , struct writeback_control * ) ;
   int (*set_page_dirty)(struct page * ) ;
   int (*readpages)(struct file * , struct address_space * , struct list_head * ,
                    unsigned int ) ;
   int (*write_begin)(struct file * , struct address_space * , loff_t , unsigned int ,
                      unsigned int , struct page ** , void ** ) ;
   int (*write_end)(struct file * , struct address_space * , loff_t , unsigned int ,
                    unsigned int , struct page * , void * ) ;
   sector_t (*bmap)(struct address_space * , sector_t ) ;
   void (*invalidatepage)(struct page * , unsigned int , unsigned int ) ;
   int (*releasepage)(struct page * , gfp_t ) ;
   void (*freepage)(struct page * ) ;
   ssize_t (*direct_IO)(int , struct kiocb * , struct iov_iter * , loff_t ) ;
   int (*get_xip_mem)(struct address_space * , unsigned long , int , void ** , unsigned long * ) ;
   int (*migratepage)(struct address_space * , struct page * , struct page * , enum migrate_mode ) ;
   int (*launder_page)(struct page * ) ;
   int (*is_partially_uptodate)(struct page * , unsigned long , unsigned long ) ;
   void (*is_dirty_writeback)(struct page * , bool * , bool * ) ;
   int (*error_remove_page)(struct address_space * , struct page * ) ;
   int (*swap_activate)(struct swap_info_struct * , struct file * , sector_t * ) ;
   void (*swap_deactivate)(struct file * ) ;
};
struct address_space {
   struct inode *host ;
   struct radix_tree_root page_tree ;
   spinlock_t tree_lock ;
   unsigned int i_mmap_writable ;
   struct rb_root i_mmap ;
   struct list_head i_mmap_nonlinear ;
   struct mutex i_mmap_mutex ;
   unsigned long nrpages ;
   unsigned long nrshadows ;
   unsigned long writeback_index ;
   struct address_space_operations const *a_ops ;
   unsigned long flags ;
   struct backing_dev_info *backing_dev_info ;
   spinlock_t private_lock ;
   struct list_head private_list ;
   void *private_data ;
};
struct request_queue;
struct hd_struct;
struct gendisk;
struct block_device {
   dev_t bd_dev ;
   int bd_openers ;
   struct inode *bd_inode ;
   struct super_block *bd_super ;
   struct mutex bd_mutex ;
   struct list_head bd_inodes ;
   void *bd_claiming ;
   void *bd_holder ;
   int bd_holders ;
   bool bd_write_holder ;
   struct list_head bd_holder_disks ;
   struct block_device *bd_contains ;
   unsigned int bd_block_size ;
   struct hd_struct *bd_part ;
   unsigned int bd_part_count ;
   int bd_invalidated ;
   struct gendisk *bd_disk ;
   struct request_queue *bd_queue ;
   struct list_head bd_list ;
   unsigned long bd_private ;
   int bd_fsfreeze_count ;
   struct mutex bd_fsfreeze_mutex ;
};
struct posix_acl;
struct inode_operations;
union __anonunion_ldv_27467_195 {
   unsigned int const i_nlink ;
   unsigned int __i_nlink ;
};
union __anonunion_ldv_27487_196 {
   struct hlist_head i_dentry ;
   struct callback_head i_rcu ;
};
struct file_lock;
struct cdev;
union __anonunion_ldv_27504_197 {
   struct pipe_inode_info *i_pipe ;
   struct block_device *i_bdev ;
   struct cdev *i_cdev ;
};
struct inode {
   umode_t i_mode ;
   unsigned short i_opflags ;
   kuid_t i_uid ;
   kgid_t i_gid ;
   unsigned int i_flags ;
   struct posix_acl *i_acl ;
   struct posix_acl *i_default_acl ;
   struct inode_operations const *i_op ;
   struct super_block *i_sb ;
   struct address_space *i_mapping ;
   void *i_security ;
   unsigned long i_ino ;
   union __anonunion_ldv_27467_195 ldv_27467 ;
   dev_t i_rdev ;
   loff_t i_size ;
   struct timespec i_atime ;
   struct timespec i_mtime ;
   struct timespec i_ctime ;
   spinlock_t i_lock ;
   unsigned short i_bytes ;
   unsigned int i_blkbits ;
   blkcnt_t i_blocks ;
   unsigned long i_state ;
   struct mutex i_mutex ;
   unsigned long dirtied_when ;
   struct hlist_node i_hash ;
   struct list_head i_wb_list ;
   struct list_head i_lru ;
   struct list_head i_sb_list ;
   union __anonunion_ldv_27487_196 ldv_27487 ;
   u64 i_version ;
   atomic_t i_count ;
   atomic_t i_dio_count ;
   atomic_t i_writecount ;
   atomic_t i_readcount ;
   struct file_operations const *i_fop ;
   struct file_lock *i_flock ;
   struct address_space i_data ;
   struct dquot *i_dquot[2U] ;
   struct list_head i_devices ;
   union __anonunion_ldv_27504_197 ldv_27504 ;
   __u32 i_generation ;
   __u32 i_fsnotify_mask ;
   struct hlist_head i_fsnotify_marks ;
   void *i_private ;
};
struct fown_struct {
   rwlock_t lock ;
   struct pid *pid ;
   enum pid_type pid_type ;
   kuid_t uid ;
   kuid_t euid ;
   int signum ;
};
struct file_ra_state {
   unsigned long start ;
   unsigned int size ;
   unsigned int async_size ;
   unsigned int ra_pages ;
   unsigned int mmap_miss ;
   loff_t prev_pos ;
};
union __anonunion_f_u_198 {
   struct llist_node fu_llist ;
   struct callback_head fu_rcuhead ;
};
struct file {
   union __anonunion_f_u_198 f_u ;
   struct path f_path ;
   struct inode *f_inode ;
   struct file_operations const *f_op ;
   spinlock_t f_lock ;
   atomic_long_t f_count ;
   unsigned int f_flags ;
   fmode_t f_mode ;
   struct mutex f_pos_lock ;
   loff_t f_pos ;
   struct fown_struct f_owner ;
   struct cred const *f_cred ;
   struct file_ra_state f_ra ;
   u64 f_version ;
   void *f_security ;
   void *private_data ;
   struct list_head f_ep_links ;
   struct list_head f_tfile_llink ;
   struct address_space *f_mapping ;
};
typedef struct files_struct *fl_owner_t;
struct file_lock_operations {
   void (*fl_copy_lock)(struct file_lock * , struct file_lock * ) ;
   void (*fl_release_private)(struct file_lock * ) ;
};
struct lock_manager_operations {
   int (*lm_compare_owner)(struct file_lock * , struct file_lock * ) ;
   unsigned long (*lm_owner_key)(struct file_lock * ) ;
   void (*lm_notify)(struct file_lock * ) ;
   int (*lm_grant)(struct file_lock * , struct file_lock * , int ) ;
   void (*lm_break)(struct file_lock * ) ;
   int (*lm_change)(struct file_lock ** , int ) ;
};
struct nlm_lockowner;
struct nfs_lock_info {
   u32 state ;
   struct nlm_lockowner *owner ;
   struct list_head list ;
};
struct nfs4_lock_state;
struct nfs4_lock_info {
   struct nfs4_lock_state *owner ;
};
struct fasync_struct;
struct __anonstruct_afs_200 {
   struct list_head link ;
   int state ;
};
union __anonunion_fl_u_199 {
   struct nfs_lock_info nfs_fl ;
   struct nfs4_lock_info nfs4_fl ;
   struct __anonstruct_afs_200 afs ;
};
struct file_lock {
   struct file_lock *fl_next ;
   struct hlist_node fl_link ;
   struct list_head fl_block ;
   fl_owner_t fl_owner ;
   unsigned int fl_flags ;
   unsigned char fl_type ;
   unsigned int fl_pid ;
   int fl_link_cpu ;
   struct pid *fl_nspid ;
   wait_queue_head_t fl_wait ;
   struct file *fl_file ;
   loff_t fl_start ;
   loff_t fl_end ;
   struct fasync_struct *fl_fasync ;
   unsigned long fl_break_time ;
   unsigned long fl_downgrade_time ;
   struct file_lock_operations const *fl_ops ;
   struct lock_manager_operations const *fl_lmops ;
   union __anonunion_fl_u_199 fl_u ;
};
struct fasync_struct {
   spinlock_t fa_lock ;
   int magic ;
   int fa_fd ;
   struct fasync_struct *fa_next ;
   struct file *fa_file ;
   struct callback_head fa_rcu ;
};
struct sb_writers {
   struct percpu_counter counter[3U] ;
   wait_queue_head_t wait ;
   int frozen ;
   wait_queue_head_t wait_unfrozen ;
   struct lockdep_map lock_map[3U] ;
};
struct super_operations;
struct xattr_handler;
struct mtd_info;
struct super_block {
   struct list_head s_list ;
   dev_t s_dev ;
   unsigned char s_blocksize_bits ;
   unsigned long s_blocksize ;
   loff_t s_maxbytes ;
   struct file_system_type *s_type ;
   struct super_operations const *s_op ;
   struct dquot_operations const *dq_op ;
   struct quotactl_ops const *s_qcop ;
   struct export_operations const *s_export_op ;
   unsigned long s_flags ;
   unsigned long s_magic ;
   struct dentry *s_root ;
   struct rw_semaphore s_umount ;
   int s_count ;
   atomic_t s_active ;
   void *s_security ;
   struct xattr_handler const **s_xattr ;
   struct list_head s_inodes ;
   struct hlist_bl_head s_anon ;
   struct list_head s_mounts ;
   struct block_device *s_bdev ;
   struct backing_dev_info *s_bdi ;
   struct mtd_info *s_mtd ;
   struct hlist_node s_instances ;
   struct quota_info s_dquot ;
   struct sb_writers s_writers ;
   char s_id[32U] ;
   u8 s_uuid[16U] ;
   void *s_fs_info ;
   unsigned int s_max_links ;
   fmode_t s_mode ;
   u32 s_time_gran ;
   struct mutex s_vfs_rename_mutex ;
   char *s_subtype ;
   char *s_options ;
   struct dentry_operations const *s_d_op ;
   int cleancache_poolid ;
   struct shrinker s_shrink ;
   atomic_long_t s_remove_count ;
   int s_readonly_remount ;
   struct workqueue_struct *s_dio_done_wq ;
   struct list_lru s_dentry_lru ;
   struct list_lru s_inode_lru ;
   struct callback_head rcu ;
};
struct fiemap_extent_info {
   unsigned int fi_flags ;
   unsigned int fi_extents_mapped ;
   unsigned int fi_extents_max ;
   struct fiemap_extent *fi_extents_start ;
};
struct dir_context {
   int (*actor)(void * , char const * , int , loff_t , u64 , unsigned int ) ;
   loff_t pos ;
};
struct file_operations {
   struct module *owner ;
   loff_t (*llseek)(struct file * , loff_t , int ) ;
   ssize_t (*read)(struct file * , char * , size_t , loff_t * ) ;
   ssize_t (*write)(struct file * , char const * , size_t , loff_t * ) ;
   ssize_t (*aio_read)(struct kiocb * , struct iovec const * , unsigned long ,
                       loff_t ) ;
   ssize_t (*aio_write)(struct kiocb * , struct iovec const * , unsigned long ,
                        loff_t ) ;
   ssize_t (*read_iter)(struct kiocb * , struct iov_iter * ) ;
   ssize_t (*write_iter)(struct kiocb * , struct iov_iter * ) ;
   int (*iterate)(struct file * , struct dir_context * ) ;
   unsigned int (*poll)(struct file * , struct poll_table_struct * ) ;
   long (*unlocked_ioctl)(struct file * , unsigned int , unsigned long ) ;
   long (*compat_ioctl)(struct file * , unsigned int , unsigned long ) ;
   int (*mmap)(struct file * , struct vm_area_struct * ) ;
   int (*open)(struct inode * , struct file * ) ;
   int (*flush)(struct file * , fl_owner_t ) ;
   int (*release)(struct inode * , struct file * ) ;
   int (*fsync)(struct file * , loff_t , loff_t , int ) ;
   int (*aio_fsync)(struct kiocb * , int ) ;
   int (*fasync)(int , struct file * , int ) ;
   int (*lock)(struct file * , int , struct file_lock * ) ;
   ssize_t (*sendpage)(struct file * , struct page * , int , size_t , loff_t * ,
                       int ) ;
   unsigned long (*get_unmapped_area)(struct file * , unsigned long , unsigned long ,
                                      unsigned long , unsigned long ) ;
   int (*check_flags)(int ) ;
   int (*flock)(struct file * , int , struct file_lock * ) ;
   ssize_t (*splice_write)(struct pipe_inode_info * , struct file * , loff_t * , size_t ,
                           unsigned int ) ;
   ssize_t (*splice_read)(struct file * , loff_t * , struct pipe_inode_info * , size_t ,
                          unsigned int ) ;
   int (*setlease)(struct file * , long , struct file_lock ** ) ;
   long (*fallocate)(struct file * , int , loff_t , loff_t ) ;
   int (*show_fdinfo)(struct seq_file * , struct file * ) ;
};
struct inode_operations {
   struct dentry *(*lookup)(struct inode * , struct dentry * , unsigned int ) ;
   void *(*follow_link)(struct dentry * , struct nameidata * ) ;
   int (*permission)(struct inode * , int ) ;
   struct posix_acl *(*get_acl)(struct inode * , int ) ;
   int (*readlink)(struct dentry * , char * , int ) ;
   void (*put_link)(struct dentry * , struct nameidata * , void * ) ;
   int (*create)(struct inode * , struct dentry * , umode_t , bool ) ;
   int (*link)(struct dentry * , struct inode * , struct dentry * ) ;
   int (*unlink)(struct inode * , struct dentry * ) ;
   int (*symlink)(struct inode * , struct dentry * , char const * ) ;
   int (*mkdir)(struct inode * , struct dentry * , umode_t ) ;
   int (*rmdir)(struct inode * , struct dentry * ) ;
   int (*mknod)(struct inode * , struct dentry * , umode_t , dev_t ) ;
   int (*rename)(struct inode * , struct dentry * , struct inode * , struct dentry * ) ;
   int (*rename2)(struct inode * , struct dentry * , struct inode * , struct dentry * ,
                  unsigned int ) ;
   int (*setattr)(struct dentry * , struct iattr * ) ;
   int (*getattr)(struct vfsmount * , struct dentry * , struct kstat * ) ;
   int (*setxattr)(struct dentry * , char const * , void const * , size_t , int ) ;
   ssize_t (*getxattr)(struct dentry * , char const * , void * , size_t ) ;
   ssize_t (*listxattr)(struct dentry * , char * , size_t ) ;
   int (*removexattr)(struct dentry * , char const * ) ;
   int (*fiemap)(struct inode * , struct fiemap_extent_info * , u64 , u64 ) ;
   int (*update_time)(struct inode * , struct timespec * , int ) ;
   int (*atomic_open)(struct inode * , struct dentry * , struct file * , unsigned int ,
                      umode_t , int * ) ;
   int (*tmpfile)(struct inode * , struct dentry * , umode_t ) ;
   int (*set_acl)(struct inode * , struct posix_acl * , int ) ;
};
struct super_operations {
   struct inode *(*alloc_inode)(struct super_block * ) ;
   void (*destroy_inode)(struct inode * ) ;
   void (*dirty_inode)(struct inode * , int ) ;
   int (*write_inode)(struct inode * , struct writeback_control * ) ;
   int (*drop_inode)(struct inode * ) ;
   void (*evict_inode)(struct inode * ) ;
   void (*put_super)(struct super_block * ) ;
   int (*sync_fs)(struct super_block * , int ) ;
   int (*freeze_fs)(struct super_block * ) ;
   int (*unfreeze_fs)(struct super_block * ) ;
   int (*statfs)(struct dentry * , struct kstatfs * ) ;
   int (*remount_fs)(struct super_block * , int * , char * ) ;
   void (*umount_begin)(struct super_block * ) ;
   int (*show_options)(struct seq_file * , struct dentry * ) ;
   int (*show_devname)(struct seq_file * , struct dentry * ) ;
   int (*show_path)(struct seq_file * , struct dentry * ) ;
   int (*show_stats)(struct seq_file * , struct dentry * ) ;
   ssize_t (*quota_read)(struct super_block * , int , char * , size_t , loff_t ) ;
   ssize_t (*quota_write)(struct super_block * , int , char const * , size_t ,
                          loff_t ) ;
   int (*bdev_try_to_free_page)(struct super_block * , struct page * , gfp_t ) ;
   long (*nr_cached_objects)(struct super_block * , int ) ;
   long (*free_cached_objects)(struct super_block * , long , int ) ;
};
struct file_system_type {
   char const *name ;
   int fs_flags ;
   struct dentry *(*mount)(struct file_system_type * , int , char const * , void * ) ;
   void (*kill_sb)(struct super_block * ) ;
   struct module *owner ;
   struct file_system_type *next ;
   struct hlist_head fs_supers ;
   struct lock_class_key s_lock_key ;
   struct lock_class_key s_umount_key ;
   struct lock_class_key s_vfs_rename_key ;
   struct lock_class_key s_writers_key[3U] ;
   struct lock_class_key i_lock_key ;
   struct lock_class_key i_mutex_key ;
   struct lock_class_key i_mutex_dir_key ;
};
struct usb_device;
struct wusb_dev;
struct ep_device;
struct usb_host_endpoint {
   struct usb_endpoint_descriptor desc ;
   struct usb_ss_ep_comp_descriptor ss_ep_comp ;
   struct list_head urb_list ;
   void *hcpriv ;
   struct ep_device *ep_dev ;
   unsigned char *extra ;
   int extralen ;
   int enabled ;
   int streams ;
};
struct usb_host_interface {
   struct usb_interface_descriptor desc ;
   int extralen ;
   unsigned char *extra ;
   struct usb_host_endpoint *endpoint ;
   char *string ;
};
enum usb_interface_condition {
    USB_INTERFACE_UNBOUND = 0,
    USB_INTERFACE_BINDING = 1,
    USB_INTERFACE_BOUND = 2,
    USB_INTERFACE_UNBINDING = 3
} ;
struct usb_interface {
   struct usb_host_interface *altsetting ;
   struct usb_host_interface *cur_altsetting ;
   unsigned int num_altsetting ;
   struct usb_interface_assoc_descriptor *intf_assoc ;
   int minor ;
   enum usb_interface_condition condition ;
   unsigned char sysfs_files_created : 1 ;
   unsigned char ep_devs_created : 1 ;
   unsigned char unregistering : 1 ;
   unsigned char needs_remote_wakeup : 1 ;
   unsigned char needs_altsetting0 : 1 ;
   unsigned char needs_binding : 1 ;
   unsigned char reset_running : 1 ;
   unsigned char resetting_device : 1 ;
   struct device dev ;
   struct device *usb_dev ;
   atomic_t pm_usage_cnt ;
   struct work_struct reset_ws ;
};
struct usb_interface_cache {
   unsigned int num_altsetting ;
   struct kref ref ;
   struct usb_host_interface altsetting[0U] ;
};
struct usb_host_config {
   struct usb_config_descriptor desc ;
   char *string ;
   struct usb_interface_assoc_descriptor *intf_assoc[16U] ;
   struct usb_interface *interface[32U] ;
   struct usb_interface_cache *intf_cache[32U] ;
   unsigned char *extra ;
   int extralen ;
};
struct usb_host_bos {
   struct usb_bos_descriptor *desc ;
   struct usb_ext_cap_descriptor *ext_cap ;
   struct usb_ss_cap_descriptor *ss_cap ;
   struct usb_ss_container_id_descriptor *ss_id ;
};
struct usb_devmap {
   unsigned long devicemap[2U] ;
};
struct mon_bus;
struct usb_bus {
   struct device *controller ;
   int busnum ;
   char const *bus_name ;
   u8 uses_dma ;
   u8 uses_pio_for_control ;
   u8 otg_port ;
   unsigned char is_b_host : 1 ;
   unsigned char b_hnp_enable : 1 ;
   unsigned char no_stop_on_short : 1 ;
   unsigned char no_sg_constraint : 1 ;
   unsigned int sg_tablesize ;
   int devnum_next ;
   struct usb_devmap devmap ;
   struct usb_device *root_hub ;
   struct usb_bus *hs_companion ;
   struct list_head bus_list ;
   struct mutex usb_address0_mutex ;
   int bandwidth_allocated ;
   int bandwidth_int_reqs ;
   int bandwidth_isoc_reqs ;
   unsigned int resuming_ports ;
   struct mon_bus *mon_bus ;
   int monitored ;
};
struct usb_tt;
enum usb_device_removable {
    USB_DEVICE_REMOVABLE_UNKNOWN = 0,
    USB_DEVICE_REMOVABLE = 1,
    USB_DEVICE_FIXED = 2
} ;
struct usb2_lpm_parameters {
   unsigned int besl ;
   int timeout ;
};
struct usb3_lpm_parameters {
   unsigned int mel ;
   unsigned int pel ;
   unsigned int sel ;
   int timeout ;
};
struct usb_device {
   int devnum ;
   char devpath[16U] ;
   u32 route ;
   enum usb_device_state state ;
   enum usb_device_speed speed ;
   struct usb_tt *tt ;
   int ttport ;
   unsigned int toggle[2U] ;
   struct usb_device *parent ;
   struct usb_bus *bus ;
   struct usb_host_endpoint ep0 ;
   struct device dev ;
   struct usb_device_descriptor descriptor ;
   struct usb_host_bos *bos ;
   struct usb_host_config *config ;
   struct usb_host_config *actconfig ;
   struct usb_host_endpoint *ep_in[16U] ;
   struct usb_host_endpoint *ep_out[16U] ;
   char **rawdescriptors ;
   unsigned short bus_mA ;
   u8 portnum ;
   u8 level ;
   unsigned char can_submit : 1 ;
   unsigned char persist_enabled : 1 ;
   unsigned char have_langid : 1 ;
   unsigned char authorized : 1 ;
   unsigned char authenticated : 1 ;
   unsigned char wusb : 1 ;
   unsigned char lpm_capable : 1 ;
   unsigned char usb2_hw_lpm_capable : 1 ;
   unsigned char usb2_hw_lpm_besl_capable : 1 ;
   unsigned char usb2_hw_lpm_enabled : 1 ;
   unsigned char usb2_hw_lpm_allowed : 1 ;
   unsigned char usb3_lpm_enabled : 1 ;
   int string_langid ;
   char *product ;
   char *manufacturer ;
   char *serial ;
   struct list_head filelist ;
   int maxchild ;
   u32 quirks ;
   atomic_t urbnum ;
   unsigned long active_duration ;
   unsigned long connect_time ;
   unsigned char do_remote_wakeup : 1 ;
   unsigned char reset_resume : 1 ;
   unsigned char port_is_suspended : 1 ;
   struct wusb_dev *wusb_dev ;
   int slot_id ;
   enum usb_device_removable removable ;
   struct usb2_lpm_parameters l1_params ;
   struct usb3_lpm_parameters u1_params ;
   struct usb3_lpm_parameters u2_params ;
   unsigned int lpm_disable_count ;
};
struct usb_iso_packet_descriptor {
   unsigned int offset ;
   unsigned int length ;
   unsigned int actual_length ;
   int status ;
};
struct usb_anchor {
   struct list_head urb_list ;
   wait_queue_head_t wait ;
   spinlock_t lock ;
   atomic_t suspend_wakeups ;
   unsigned char poisoned : 1 ;
};
struct urb {
   struct kref kref ;
   void *hcpriv ;
   atomic_t use_count ;
   atomic_t reject ;
   int unlinked ;
   struct list_head urb_list ;
   struct list_head anchor_list ;
   struct usb_anchor *anchor ;
   struct usb_device *dev ;
   struct usb_host_endpoint *ep ;
   unsigned int pipe ;
   unsigned int stream_id ;
   int status ;
   unsigned int transfer_flags ;
   void *transfer_buffer ;
   dma_addr_t transfer_dma ;
   struct scatterlist *sg ;
   int num_mapped_sgs ;
   int num_sgs ;
   u32 transfer_buffer_length ;
   u32 actual_length ;
   unsigned char *setup_packet ;
   dma_addr_t setup_dma ;
   int start_frame ;
   int number_of_packets ;
   int interval ;
   int error_count ;
   void *context ;
   void (*complete)(struct urb * ) ;
   struct usb_iso_packet_descriptor iso_frame_desc[0U] ;
};
struct giveback_urb_bh {
   bool running ;
   spinlock_t lock ;
   struct list_head head ;
   struct tasklet_struct bh ;
   struct usb_host_endpoint *completing_ep ;
};
struct hc_driver;
struct usb_phy;
struct dma_pool;
struct usb_hcd {
   struct usb_bus self ;
   struct kref kref ;
   char const *product_desc ;
   int speed ;
   char irq_descr[24U] ;
   struct timer_list rh_timer ;
   struct urb *status_urb ;
   struct work_struct wakeup_work ;
   struct hc_driver const *driver ;
   struct usb_phy *phy ;
   unsigned long flags ;
   unsigned char rh_registered : 1 ;
   unsigned char rh_pollable : 1 ;
   unsigned char msix_enabled : 1 ;
   unsigned char remove_phy : 1 ;
   unsigned char uses_new_polling : 1 ;
   unsigned char wireless : 1 ;
   unsigned char authorized_default : 1 ;
   unsigned char has_tt : 1 ;
   unsigned char amd_resume_bug : 1 ;
   unsigned char can_do_streams : 1 ;
   unsigned int irq ;
   void *regs ;
   resource_size_t rsrc_start ;
   resource_size_t rsrc_len ;
   unsigned int power_budget ;
   struct giveback_urb_bh high_prio_bh ;
   struct giveback_urb_bh low_prio_bh ;
   struct mutex *bandwidth_mutex ;
   struct usb_hcd *shared_hcd ;
   struct usb_hcd *primary_hcd ;
   struct dma_pool *pool[4U] ;
   int state ;
   unsigned long hcd_priv[0U] ;
};
struct hc_driver {
   char const *description ;
   char const *product_desc ;
   size_t hcd_priv_size ;
   irqreturn_t (*irq)(struct usb_hcd * ) ;
   int flags ;
   int (*reset)(struct usb_hcd * ) ;
   int (*start)(struct usb_hcd * ) ;
   int (*pci_suspend)(struct usb_hcd * , bool ) ;
   int (*pci_resume)(struct usb_hcd * , bool ) ;
   void (*stop)(struct usb_hcd * ) ;
   void (*shutdown)(struct usb_hcd * ) ;
   int (*get_frame_number)(struct usb_hcd * ) ;
   int (*urb_enqueue)(struct usb_hcd * , struct urb * , gfp_t ) ;
   int (*urb_dequeue)(struct usb_hcd * , struct urb * , int ) ;
   int (*map_urb_for_dma)(struct usb_hcd * , struct urb * , gfp_t ) ;
   void (*unmap_urb_for_dma)(struct usb_hcd * , struct urb * ) ;
   void (*endpoint_disable)(struct usb_hcd * , struct usb_host_endpoint * ) ;
   void (*endpoint_reset)(struct usb_hcd * , struct usb_host_endpoint * ) ;
   int (*hub_status_data)(struct usb_hcd * , char * ) ;
   int (*hub_control)(struct usb_hcd * , u16 , u16 , u16 , char * , u16 ) ;
   int (*bus_suspend)(struct usb_hcd * ) ;
   int (*bus_resume)(struct usb_hcd * ) ;
   int (*start_port_reset)(struct usb_hcd * , unsigned int ) ;
   void (*relinquish_port)(struct usb_hcd * , int ) ;
   int (*port_handed_over)(struct usb_hcd * , int ) ;
   void (*clear_tt_buffer_complete)(struct usb_hcd * , struct usb_host_endpoint * ) ;
   int (*alloc_dev)(struct usb_hcd * , struct usb_device * ) ;
   void (*free_dev)(struct usb_hcd * , struct usb_device * ) ;
   int (*alloc_streams)(struct usb_hcd * , struct usb_device * , struct usb_host_endpoint ** ,
                        unsigned int , unsigned int , gfp_t ) ;
   int (*free_streams)(struct usb_hcd * , struct usb_device * , struct usb_host_endpoint ** ,
                       unsigned int , gfp_t ) ;
   int (*add_endpoint)(struct usb_hcd * , struct usb_device * , struct usb_host_endpoint * ) ;
   int (*drop_endpoint)(struct usb_hcd * , struct usb_device * , struct usb_host_endpoint * ) ;
   int (*check_bandwidth)(struct usb_hcd * , struct usb_device * ) ;
   void (*reset_bandwidth)(struct usb_hcd * , struct usb_device * ) ;
   int (*address_device)(struct usb_hcd * , struct usb_device * ) ;
   int (*enable_device)(struct usb_hcd * , struct usb_device * ) ;
   int (*update_hub_device)(struct usb_hcd * , struct usb_device * , struct usb_tt * ,
                            gfp_t ) ;
   int (*reset_device)(struct usb_hcd * , struct usb_device * ) ;
   int (*update_device)(struct usb_hcd * , struct usb_device * ) ;
   int (*set_usb2_hw_lpm)(struct usb_hcd * , struct usb_device * , int ) ;
   int (*enable_usb3_lpm_timeout)(struct usb_hcd * , struct usb_device * , enum usb3_link_state ) ;
   int (*disable_usb3_lpm_timeout)(struct usb_hcd * , struct usb_device * , enum usb3_link_state ) ;
   int (*find_raw_port_number)(struct usb_hcd * , int ) ;
};
struct __anonstruct_hs_202 {
   __u8 DeviceRemovable[4U] ;
   __u8 PortPwrCtrlMask[4U] ;
};
struct __anonstruct_ss_203 {
   __u8 bHubHdrDecLat ;
   __le16 wHubDelay ;
   __le16 DeviceRemovable ;
};
union __anonunion_u_201 {
   struct __anonstruct_hs_202 hs ;
   struct __anonstruct_ss_203 ss ;
};
struct usb_hub_descriptor {
   __u8 bDescLength ;
   __u8 bDescriptorType ;
   __u8 bNbrPorts ;
   __le16 wHubCharacteristics ;
   __u8 bPwrOn2PwrGood ;
   __u8 bHubContrCurrent ;
   union __anonunion_u_201 u ;
};
struct usb_tt {
   struct usb_device *hub ;
   int multi ;
   unsigned int think_time ;
   void *hcpriv ;
   spinlock_t lock ;
   struct list_head clear_list ;
   struct work_struct clear_work ;
};
struct max3421_hcd_platform_data {
   u8 vbus_gpout ;
   u8 vbus_active_level ;
};
enum max3421_rh_state {
    MAX3421_RH_RESET = 0,
    MAX3421_RH_SUSPENDED = 1,
    MAX3421_RH_RUNNING = 2
} ;
enum pkt_state {
    PKT_STATE_SETUP = 0,
    PKT_STATE_TRANSFER = 1,
    PKT_STATE_TERMINATE = 2
} ;
enum scheduling_pass {
    SCHED_PASS_PERIODIC = 0,
    SCHED_PASS_NON_PERIODIC = 1,
    SCHED_PASS_DONE = 2
} ;
struct max3421_dma_buf {
   u8 data[2U] ;
};
struct max3421_hcd {
   spinlock_t lock ;
   struct task_struct *spi_thread ;
   struct max3421_hcd *next ;
   enum max3421_rh_state rh_state ;
   u32 port_status ;
   unsigned char active : 1 ;
   struct list_head ep_list ;
   u8 rev ;
   u16 frame_number ;
   struct max3421_dma_buf *tx ;
   struct max3421_dma_buf *rx ;
   struct urb *curr_urb ;
   enum scheduling_pass sched_pass ;
   struct usb_device *loaded_dev ;
   int loaded_epnum ;
   int urb_done ;
   size_t curr_len ;
   u8 hien ;
   u8 mode ;
   u8 iopins[2U] ;
   unsigned char do_enable_irq : 1 ;
   unsigned char do_reset_hcd : 1 ;
   unsigned char do_reset_port : 1 ;
   unsigned char do_check_unlink : 1 ;
   unsigned char do_iopin_update : 1 ;
};
struct max3421_ep {
   struct usb_host_endpoint *ep ;
   struct list_head ep_list ;
   u32 naks ;
   u16 last_active ;
   enum pkt_state pkt_state ;
   u8 retries ;
   u8 retransmit ;
};
typedef int ldv_func_ret_type___2;
typedef struct page___0 *pgtable_t___0;
struct __anonstruct____missing_field_name_211 {
   unsigned int inuse : 16 ;
   unsigned int objects : 15 ;
   unsigned int frozen : 1 ;
};
union __anonunion____missing_field_name_210 {
   atomic_t _mapcount ;
   struct __anonstruct____missing_field_name_211 __annonCompField39 ;
   int units ;
};
struct __anonstruct____missing_field_name_209 {
   union __anonunion____missing_field_name_210 __annonCompField40 ;
   atomic_t _count ;
};
union __anonunion____missing_field_name_208 {
   unsigned long counters ;
   struct __anonstruct____missing_field_name_209 __annonCompField41 ;
   unsigned int active ;
};
struct __anonstruct____missing_field_name_206 {
   union __anonunion_ldv_14126_140 __annonCompField38 ;
   union __anonunion____missing_field_name_208 __annonCompField42 ;
};
struct __anonstruct____missing_field_name_213 {
   struct page___0 *next ;
   int pages ;
   int pobjects ;
};
union __anonunion____missing_field_name_212 {
   struct list_head lru ;
   struct __anonstruct____missing_field_name_213 __annonCompField44 ;
   struct slab *slab_page ;
   struct callback_head callback_head ;
   pgtable_t___0 pmd_huge_pte ;
};
union __anonunion____missing_field_name_214 {
   unsigned long private ;
   spinlock_t *ptl ;
   struct kmem_cache___0 *slab_cache ;
   struct page___0 *first_page ;
};
struct page___0 {
   unsigned long flags ;
   union __anonunion_ldv_14120_138 __annonCompField37 ;
   struct __anonstruct____missing_field_name_206 __annonCompField43 ;
   union __anonunion____missing_field_name_212 __annonCompField45 ;
   union __anonunion____missing_field_name_214 __annonCompField46 ;
   unsigned long debug_flags ;
} __attribute__((__aligned__((2) * (sizeof(unsigned long )) ))) ;
enum kobj_ns_type;
struct attribute___0 {
   char const *name ;
   umode_t mode ;
   bool ignore_lockdep : 1 ;
   struct lock_class_key *key ;
   struct lock_class_key skey ;
};
struct sysfs_ops___0 {
   ssize_t (*show)(struct kobject___0 * , struct attribute___0 * , char * ) ;
   ssize_t (*store)(struct kobject___0 * , struct attribute___0 * , char const * ,
                    size_t ) ;
};
struct kobject___0 {
   char const *name ;
   struct list_head entry ;
   struct kobject___0 *parent ;
   struct kset *kset ;
   struct kobj_type___0 *ktype ;
   struct kernfs_node *sd ;
   struct kref kref ;
   struct delayed_work release ;
   unsigned int state_initialized : 1 ;
   unsigned int state_in_sysfs : 1 ;
   unsigned int state_add_uevent_sent : 1 ;
   unsigned int state_remove_uevent_sent : 1 ;
   unsigned int uevent_suppress : 1 ;
};
struct kobj_type___0 {
   void (*release)(struct kobject___0 *kobj ) ;
   struct sysfs_ops___0 const *sysfs_ops ;
   struct attribute___0 **default_attrs ;
   struct kobj_ns_type_operations const *(*child_ns_type)(struct kobject___0 *kobj ) ;
   void const *(*namespace)(struct kobject___0 *kobj ) ;
};
struct kmem_cache_cpu___0 {
   void **freelist ;
   unsigned long tid ;
   struct page___0 *page ;
   struct page___0 *partial ;
   unsigned int stat[26] ;
};
struct kmem_cache___0 {
   struct kmem_cache_cpu___0 *cpu_slab ;
   unsigned long flags ;
   unsigned long min_partial ;
   int size ;
   int object_size ;
   int offset ;
   int cpu_partial ;
   struct kmem_cache_order_objects oo ;
   struct kmem_cache_order_objects max ;
   struct kmem_cache_order_objects min ;
   gfp_t allocflags ;
   int refcount ;
   void (*ctor)(void * ) ;
   int inuse ;
   int align ;
   int reserved ;
   char const *name ;
   struct list_head list ;
   struct kobject___0 kobj ;
   struct memcg_cache_params___0 *memcg_params ;
   int max_attr_size ;
   struct kset *memcg_kset ;
   int remote_node_defrag_ratio ;
   struct kmem_cache_node *node[1 << 10] ;
};
struct __anonstruct____missing_field_name_227 {
   struct callback_head callback_head ;
   struct kmem_cache___0 *memcg_caches[0] ;
};
struct __anonstruct____missing_field_name_228 {
   struct mem_cgroup *memcg ;
   struct list_head list ;
   struct kmem_cache___0 *root_cache ;
   atomic_t nr_pages ;
};
union __anonunion____missing_field_name_226 {
   struct __anonstruct____missing_field_name_227 __annonCompField50 ;
   struct __anonstruct____missing_field_name_228 __annonCompField51 ;
};
struct memcg_cache_params___0 {
   bool is_root_cache ;
   union __anonunion____missing_field_name_226 __annonCompField52 ;
};
long ldv__builtin_expect(long exp , long c ) ;
void ldv_spin_lock(void) ;
void ldv_spin_unlock(void) ;
extern struct module __this_module ;
__inline static void set_bit(long nr , unsigned long volatile *addr )
{
  {
  __asm__ volatile (".pushsection .smp_locks,\"a\"\n.balign 4\n.long 671f - .\n.popsection\n671:\n\tlock; bts %1,%0": "+m" (*((long volatile *)addr)): "Ir" (nr): "memory");
  return;
}
}
extern int printk(char const * , ...) ;
extern int __dynamic_dev_dbg(struct _ddebug * , struct device const * , char const *
                             , ...) ;
extern void __might_sleep(char const * , int , int ) ;
__inline static void INIT_LIST_HEAD(struct list_head *list )
{
  {
  list->next = list;
  list->prev = list;
  return;
}
}
extern void __list_add(struct list_head * , struct list_head * , struct list_head * ) ;
__inline static void list_add_tail(struct list_head *new , struct list_head *head )
{
  {
  __list_add(new, head->prev, head);
  return;
}
}
extern void __list_del_entry(struct list_head * ) ;
extern void list_del(struct list_head * ) ;
__inline static void list_move_tail(struct list_head *list , struct list_head *head )
{
  {
  __list_del_entry(list);
  list_add_tail(list, head);
  return;
}
}
__inline static int list_empty(struct list_head const *head )
{
  {
  return ((unsigned long )((struct list_head const *)head->next) == (unsigned long )head);
}
}
extern void __bad_percpu_size(void) ;
extern void warn_slowpath_null(char const * , int const ) ;
extern struct task_struct *current_task ;
__inline static struct task_struct *get_current(void)
{
  struct task_struct *pfo_ret__ ;
  {
  switch (8UL) {
  case 1UL:
  __asm__ ("movb %%gs:%P1,%0": "=q" (pfo_ret__): "p" (& current_task));
  goto ldv_3067;
  case 2UL:
  __asm__ ("movw %%gs:%P1,%0": "=r" (pfo_ret__): "p" (& current_task));
  goto ldv_3067;
  case 4UL:
  __asm__ ("movl %%gs:%P1,%0": "=r" (pfo_ret__): "p" (& current_task));
  goto ldv_3067;
  case 8UL:
  __asm__ ("movq %%gs:%P1,%0": "=r" (pfo_ret__): "p" (& current_task));
  goto ldv_3067;
  default:
  __bad_percpu_size();
  }
  ldv_3067: ;
  return (pfo_ret__);
}
}
extern void *memset(void * , int , size_t ) ;
__inline static void *ERR_PTR(long error )
{
  {
  return ((void *)error);
}
}
__inline static bool IS_ERR(void const *ptr )
{
  long tmp ;
  {
  tmp = ldv__builtin_expect((unsigned long )ptr > 0xfffffffffffff000UL, 0L);
  return (tmp != 0L);
}
}
extern void __xchg_wrong_size(void) ;
extern void __raw_spin_lock_init(raw_spinlock_t * , char const * , struct lock_class_key * ) ;
extern void _raw_spin_unlock_irqrestore(raw_spinlock_t * , unsigned long ) ;
__inline static raw_spinlock_t *spinlock_check(spinlock_t *lock )
{
  {
  return (& lock->ldv_6347.rlock);
}
}
__inline static void ldv_spin_unlock_irqrestore_8(spinlock_t *lock , unsigned long flags )
{
  {
  _raw_spin_unlock_irqrestore(& lock->ldv_6347.rlock, flags);
  return;
}
}
__inline static void spin_unlock_irqrestore(spinlock_t *lock , unsigned long flags ) ;
__inline static char const *kobject_name(struct kobject const *kobj )
{
  {
  return ((char const *)kobj->name);
}
}
extern void kfree(void const * ) ;
extern void *ldv_malloc(size_t);
void *__kmalloc(size_t size, gfp_t t)
{
 return ldv_malloc(size);
}
extern void *kmem_cache_alloc(struct kmem_cache * , gfp_t ) ;
__inline static void *ldv_kmalloc_12(size_t size , gfp_t flags )
{
  void *tmp___2 ;
  {
  tmp___2 = __kmalloc(size, flags);
  return (tmp___2);
}
}
__inline static void *kmalloc(size_t size , gfp_t flags ) ;
__inline static void *kzalloc(size_t size , gfp_t flags ) ;
void ldv_check_alloc_flags(gfp_t flags ) ;
extern void *malloc(size_t size ) ;
extern void *calloc(size_t nmemb , size_t size ) ;
extern int __VERIFIER_nondet_int(void) ;
extern u16 __VERIFIER_nondet_u16(void) ;
extern unsigned long __VERIFIER_nondet_ulong(void) ;
extern void abort(void);
void assume_abort_if_not(int cond) {
  if(!cond) {abort();}
}

void *ldv_malloc(size_t size )
{
  void *p ;
  void *tmp ;
  int tmp___0 ;
  {
  tmp___0 = __VERIFIER_nondet_int();
  if (tmp___0 != 0) {
    return ((void *)0);
  } else {
    tmp = malloc(size);
    p = tmp;
    assume_abort_if_not((unsigned long )p != (unsigned long )((void *)0));
    assume_abort_if_not(IS_ERR(p) == 0);
    return (p);
  }
}
}
void *ldv_zalloc(size_t size )
{
  void *p ;
  void *tmp ;
  int tmp___0 ;
  {
  tmp___0 = __VERIFIER_nondet_int();
  if (tmp___0 != 0) {
    return ((void *)0);
  } else {
    tmp = calloc(1UL, size);
    p = tmp;
    assume_abort_if_not((unsigned long )p != (unsigned long )((void *)0));
    assume_abort_if_not(IS_ERR(p) == 0);
    return (p);
  }
}
}
int ldv_undef_int(void)
{
  int tmp ;
  {
  tmp = __VERIFIER_nondet_int();
  return (tmp);
}
}
unsigned long ldv_undef_ulong(void)
{
  unsigned long tmp ;
  {
  tmp = __VERIFIER_nondet_ulong();
  return (tmp);
}
}
__inline static void ldv_error(void)
{
  {
  ERROR: ;
  {reach_error();}
}
}
__inline static void ldv_stop(void)
{
  {
  LDV_STOP: ;
  goto LDV_STOP;
}
}
long ldv__builtin_expect(long exp , long c )
{
  {
  return (exp);
}
}
void ldv__builtin_trap(void)
{
  {
  ldv_error();
  return;
}
}
int ldv_irq_1_2 = 0;
int LDV_IN_INTERRUPT = 1;
int ldv_irq_1_3 = 0;
void *ldv_irq_data_1_1 ;
int ldv_irq_1_1 = 0;
int ldv_irq_1_0 = 0;
struct urb *max3421_hcd_desc_group0 ;
int ldv_irq_line_1_3 ;
void *ldv_irq_data_1_0 ;
int ldv_state_variable_0 ;
struct usb_hcd *max3421_hcd_desc_group1 ;
int ldv_state_variable_3 ;
int ldv_irq_line_1_0 ;
int ldv_state_variable_2 ;
void *ldv_irq_data_1_3 ;
struct spi_device *max3421_driver_group0 ;
int ref_cnt ;
int ldv_irq_line_1_1 ;
void *ldv_irq_data_1_2 ;
int ldv_state_variable_1 ;
int ldv_irq_line_1_2 ;
int ldv_irq_1(int state , int line , void *data ) ;
void activate_suitable_irq_1(int line , void *data ) ;
int reg_check_1(irqreturn_t (*handler)(int , void * ) ) ;
void ldv_initialize_hc_driver_3(void) ;
void choose_interrupt_1(void) ;
void ldv_initialize_spi_driver_2(void) ;
void disable_suitable_irq_1(int line , void *data ) ;
extern void driver_unregister(struct device_driver * ) ;
__inline static char const *dev_name(struct device const *dev )
{
  char const *tmp ;
  {
  if ((unsigned long )dev->init_name != (unsigned long )((char const * )0)) {
    return ((char const *)dev->init_name);
  } else {
  }
  tmp = kobject_name(& dev->kobj);
  return (tmp);
}
}
extern int dev_err(struct device const * , char const * , ...) ;
extern int _dev_info(struct device const * , char const * , ...) ;
extern void schedule(void) ;
extern int wake_up_process(struct task_struct * ) ;
extern int _cond_resched(void) ;
extern struct task_struct *kthread_create_on_node(int (*)(void * ) , void * , int ,
                                                  char const * , ...) ;
extern int kthread_stop(struct task_struct * ) ;
extern bool kthread_should_stop(void) ;
__inline static struct spi_device *to_spi_device(struct device *dev )
{
  struct device const *__mptr ;
  struct spi_device *tmp ;
  {
  if ((unsigned long )dev != (unsigned long )((struct device *)0)) {
    __mptr = (struct device const *)dev;
    tmp = (struct spi_device *)__mptr;
  } else {
    tmp = (struct spi_device *)0;
  }
  return (tmp);
}
}
extern int spi_register_driver(struct spi_driver * ) ;
__inline static void spi_unregister_driver(struct spi_driver *sdrv )
{
  {
  if ((unsigned long )sdrv != (unsigned long )((struct spi_driver *)0)) {
    driver_unregister(& sdrv->driver);
  } else {
  }
  return;
}
}
__inline static void spi_message_init(struct spi_message *m )
{
  {
  memset((void *)m, 0, 88UL);
  INIT_LIST_HEAD(& m->transfers);
  return;
}
}
__inline static void spi_message_add_tail(struct spi_transfer *t , struct spi_message *m )
{
  {
  list_add_tail(& t->transfer_list, & m->transfers);
  return;
}
}
extern int spi_setup(struct spi_device * ) ;
extern int spi_sync(struct spi_device * , struct spi_message * ) ;
__inline static int usb_endpoint_num(struct usb_endpoint_descriptor const *epd )
{
  {
  return ((int )epd->bEndpointAddress & 15);
}
}
__inline static int usb_endpoint_type(struct usb_endpoint_descriptor const *epd )
{
  {
  return ((int )epd->bmAttributes & 3);
}
}
__inline static int usb_endpoint_xfer_control(struct usb_endpoint_descriptor const *epd )
{
  {
  return (((int )epd->bmAttributes & 3) == 0);
}
}
__inline static int usb_endpoint_maxp(struct usb_endpoint_descriptor const *epd )
{
  {
  return ((int )epd->wMaxPacketSize);
}
}
extern void msleep(unsigned int ) ;
extern int request_threaded_irq(unsigned int , irqreturn_t (*)(int , void * ) ,
                                irqreturn_t (*)(int , void * ) , unsigned long ,
                                char const * , void * ) ;
__inline static int request_irq(unsigned int irq , irqreturn_t (*handler)(int , void * ) ,
                                unsigned long flags , char const *name , void *dev )
{
  int tmp ;
  {
  tmp = request_threaded_irq(irq, handler, (irqreturn_t (*)(int , void * ))0, flags,
                             name, dev);
  return (tmp);
}
}
__inline static int ldv_request_irq_19(unsigned int irq , irqreturn_t (*handler)(int ,
                                                                                 void * ) ,
                                       unsigned long flags , char const *name ,
                                       void *dev ) ;
extern void free_irq(unsigned int , void * ) ;
void ldv_free_irq_20(unsigned int ldv_func_arg1 , void *ldv_func_arg2 ) ;
extern void disable_irq_nosync(unsigned int ) ;
extern void enable_irq(unsigned int ) ;
__inline static int usb_urb_dir_in(struct urb *urb )
{
  {
  return ((urb->transfer_flags & 512U) != 0U);
}
}
__inline static int usb_urb_dir_out(struct urb *urb )
{
  {
  return ((urb->transfer_flags & 512U) == 0U);
}
}
__inline static __u16 usb_maxpacket(struct usb_device *udev , int pipe , int is_out )
{
  struct usb_host_endpoint *ep ;
  unsigned int epnum ;
  int __ret_warn_on ;
  long tmp ;
  int __ret_warn_on___0 ;
  long tmp___0 ;
  int tmp___1 ;
  {
  epnum = (unsigned int )(pipe >> 15) & 15U;
  if (is_out != 0) {
    __ret_warn_on = (pipe & 128) != 0;
    tmp = ldv__builtin_expect(__ret_warn_on != 0, 0L);
    if (tmp != 0L) {
      warn_slowpath_null("include/linux/usb.h", 1825);
    } else {
    }
    ldv__builtin_expect(__ret_warn_on != 0, 0L);
    ep = udev->ep_out[epnum];
  } else {
    __ret_warn_on___0 = (pipe & 128) == 0;
    tmp___0 = ldv__builtin_expect(__ret_warn_on___0 != 0, 0L);
    if (tmp___0 != 0L) {
      warn_slowpath_null("include/linux/usb.h", 1828);
    } else {
    }
    ldv__builtin_expect(__ret_warn_on___0 != 0, 0L);
    ep = udev->ep_in[epnum];
  }
  if ((unsigned long )ep == (unsigned long )((struct usb_host_endpoint *)0)) {
    return (0U);
  } else {
  }
  tmp___1 = usb_endpoint_maxp((struct usb_endpoint_descriptor const *)(& ep->desc));
  return ((__u16 )tmp___1);
}
}
extern int usb_hcd_link_urb_to_ep(struct usb_hcd * , struct urb * ) ;
extern int usb_hcd_check_unlink_urb(struct usb_hcd * , struct urb * , int ) ;
extern void usb_hcd_unlink_urb_from_ep(struct usb_hcd * , struct urb * ) ;
extern void usb_hcd_giveback_urb(struct usb_hcd * , struct urb * , int ) ;
extern struct usb_hcd *usb_create_hcd(struct hc_driver const * , struct device * ,
                                      char const * ) ;
extern void usb_put_hcd(struct usb_hcd * ) ;
extern int usb_add_hcd(struct usb_hcd * , unsigned int , unsigned long ) ;
extern void usb_remove_hcd(struct usb_hcd * ) ;
extern void usb_hcd_resume_root_hub(struct usb_hcd * ) ;
static struct max3421_hcd *max3421_hcd_list ;
static int const hrsl_to_error[16U] =
  { 0, -22, -22, -22,
        -11, -32, -84, -71,
        -121, -71, -71, -84,
        -5, -5, -62, -75};
__inline static s16 frame_diff(u16 left , u16 right )
{
  {
  return ((int )((s16 )((int )left - (int )right)) & 2047);
}
}
__inline static struct max3421_hcd *hcd_to_max3421(struct usb_hcd *hcd )
{
  {
  return ((struct max3421_hcd *)(& hcd->hcd_priv));
}
}
__inline static struct usb_hcd *max3421_to_hcd(struct max3421_hcd *max3421_hcd )
{
  unsigned long const (*__mptr)[0U] ;
  {
  __mptr = (unsigned long const *)max3421_hcd;
  return ((struct usb_hcd *)__mptr + 0xfffffffffffffc38UL);
}
}
static u8 spi_rd8(struct usb_hcd *hcd , unsigned int reg )
{
  struct max3421_hcd *max3421_hcd ;
  struct max3421_hcd *tmp ;
  struct spi_device *spi ;
  struct spi_device *tmp___0 ;
  struct spi_transfer transfer ;
  struct spi_message msg ;
  {
  tmp = hcd_to_max3421(hcd);
  max3421_hcd = tmp;
  tmp___0 = to_spi_device(hcd->self.controller);
  spi = tmp___0;
  memset((void *)(& transfer), 0, 96UL);
  spi_message_init(& msg);
  (max3421_hcd->tx)->data[0] = (int )((u8 )reg) << 3U;
  transfer.tx_buf = (void const *)(& (max3421_hcd->tx)->data);
  transfer.rx_buf = (void *)(& (max3421_hcd->rx)->data);
  transfer.len = 2U;
  spi_message_add_tail(& transfer, & msg);
  spi_sync(spi, & msg);
  return ((max3421_hcd->rx)->data[1]);
}
}
static void spi_wr8(struct usb_hcd *hcd , unsigned int reg , u8 val )
{
  struct spi_device *spi ;
  struct spi_device *tmp ;
  struct max3421_hcd *max3421_hcd ;
  struct max3421_hcd *tmp___0 ;
  struct spi_transfer transfer ;
  struct spi_message msg ;
  {
  tmp = to_spi_device(hcd->self.controller);
  spi = tmp;
  tmp___0 = hcd_to_max3421(hcd);
  max3421_hcd = tmp___0;
  memset((void *)(& transfer), 0, 96UL);
  spi_message_init(& msg);
  (max3421_hcd->tx)->data[0] = (unsigned int )((int )((u8 )reg) << 3U) | 2U;
  (max3421_hcd->tx)->data[1] = val;
  transfer.tx_buf = (void const *)(& (max3421_hcd->tx)->data);
  transfer.len = 2U;
  spi_message_add_tail(& transfer, & msg);
  spi_sync(spi, & msg);
  return;
}
}
static void spi_rd_buf(struct usb_hcd *hcd , unsigned int reg , void *buf , size_t len )
{
  struct spi_device *spi ;
  struct spi_device *tmp ;
  struct max3421_hcd *max3421_hcd ;
  struct max3421_hcd *tmp___0 ;
  struct spi_transfer transfer[2U] ;
  struct spi_message msg ;
  {
  tmp = to_spi_device(hcd->self.controller);
  spi = tmp;
  tmp___0 = hcd_to_max3421(hcd);
  max3421_hcd = tmp___0;
  memset((void *)(& transfer), 0, 192UL);
  spi_message_init(& msg);
  (max3421_hcd->tx)->data[0] = (int )((u8 )reg) << 3U;
  transfer[0].tx_buf = (void const *)(& (max3421_hcd->tx)->data);
  transfer[0].len = 1U;
  transfer[1].rx_buf = buf;
  transfer[1].len = (unsigned int )len;
  spi_message_add_tail((struct spi_transfer *)(& transfer), & msg);
  spi_message_add_tail((struct spi_transfer *)(& transfer) + 1UL, & msg);
  spi_sync(spi, & msg);
  return;
}
}
static void spi_wr_buf(struct usb_hcd *hcd , unsigned int reg , void *buf , size_t len )
{
  struct spi_device *spi ;
  struct spi_device *tmp ;
  struct max3421_hcd *max3421_hcd ;
  struct max3421_hcd *tmp___0 ;
  struct spi_transfer transfer[2U] ;
  struct spi_message msg ;
  {
  tmp = to_spi_device(hcd->self.controller);
  spi = tmp;
  tmp___0 = hcd_to_max3421(hcd);
  max3421_hcd = tmp___0;
  memset((void *)(& transfer), 0, 192UL);
  spi_message_init(& msg);
  (max3421_hcd->tx)->data[0] = (unsigned int )((int )((u8 )reg) << 3U) | 2U;
  transfer[0].tx_buf = (void const *)(& (max3421_hcd->tx)->data);
  transfer[0].len = 1U;
  transfer[1].tx_buf = (void const *)buf;
  transfer[1].len = (unsigned int )len;
  spi_message_add_tail((struct spi_transfer *)(& transfer), & msg);
  spi_message_add_tail((struct spi_transfer *)(& transfer) + 1UL, & msg);
  spi_sync(spi, & msg);
  return;
}
}
static void max3421_set_speed(struct usb_hcd *hcd , struct usb_device *dev )
{
  struct max3421_hcd *max3421_hcd ;
  struct max3421_hcd *tmp ;
  u8 mode_lowspeed ;
  u8 mode_hubpre ;
  u8 mode ;
  {
  tmp = hcd_to_max3421(hcd);
  max3421_hcd = tmp;
  mode = max3421_hcd->mode;
  mode_lowspeed = 2U;
  mode_hubpre = 4U;
  if ((max3421_hcd->port_status & 512U) != 0U) {
    mode = (u8 )((int )mode | (int )mode_lowspeed);
    mode = (u8 )(~ ((int )((signed char )mode_hubpre)) & (int )((signed char )mode));
  } else
  if ((unsigned int )dev->speed == 1U) {
    mode = (u8 )(((int )mode_lowspeed | (int )mode_hubpre) | (int )mode);
  } else {
    mode = (u8 )((int )((signed char )(~ ((int )mode_lowspeed | (int )mode_hubpre))) & (int )((signed char )mode));
  }
  if ((int )max3421_hcd->mode != (int )mode) {
    max3421_hcd->mode = mode;
    spi_wr8(hcd, 27U, (int )max3421_hcd->mode);
  } else {
  }
  return;
}
}
static void max3421_set_address(struct usb_hcd *hcd , struct usb_device *dev , int epnum ,
                                int force_toggles )
{
  struct max3421_hcd *max3421_hcd ;
  struct max3421_hcd *tmp ;
  int old_epnum ;
  int same_ep ;
  int rcvtog ;
  int sndtog ;
  struct usb_device *old_dev ;
  u8 hctl ;
  u8 hrsl ;
  u8 tmp___0 ;
  {
  tmp = hcd_to_max3421(hcd);
  max3421_hcd = tmp;
  old_dev = max3421_hcd->loaded_dev;
  old_epnum = max3421_hcd->loaded_epnum;
  same_ep = (unsigned long )dev == (unsigned long )old_dev && epnum == old_epnum;
  if (same_ep != 0 && force_toggles == 0) {
    return;
  } else {
  }
  if ((unsigned long )old_dev != (unsigned long )((struct usb_device *)0) && same_ep == 0) {
    tmp___0 = spi_rd8(hcd, 31U);
    hrsl = tmp___0;
    rcvtog = ((int )hrsl >> 4) & 1;
    sndtog = ((int )hrsl >> 5) & 1;
    old_dev->toggle[0] = (old_dev->toggle[0] & (unsigned int )(~ (1 << old_epnum))) | (unsigned int )(rcvtog << old_epnum);
    old_dev->toggle[1] = (old_dev->toggle[1] & (unsigned int )(~ (1 << old_epnum))) | (unsigned int )(sndtog << old_epnum);
  } else {
  }
  rcvtog = (int )(dev->toggle[0] >> epnum) & 1;
  sndtog = (int )(dev->toggle[1] >> epnum) & 1;
  hctl = (int )((u8 )(1UL << (rcvtog + 4))) | (int )((u8 )(1UL << (sndtog + 6)));
  max3421_hcd->loaded_epnum = epnum;
  spi_wr8(hcd, 29U, (int )hctl);
  max3421_hcd->loaded_dev = dev;
  spi_wr8(hcd, 28U, (int )((u8 )dev->devnum));
  return;
}
}
static int max3421_ctrl_setup(struct usb_hcd *hcd , struct urb *urb )
{
  {
  spi_wr_buf(hcd, 4U, (void *)urb->setup_packet, 8UL);
  return (16);
}
}
static int max3421_transfer_in(struct usb_hcd *hcd , struct urb *urb )
{
  struct max3421_hcd *max3421_hcd ;
  struct max3421_hcd *tmp ;
  int epnum ;
  {
  tmp = hcd_to_max3421(hcd);
  max3421_hcd = tmp;
  epnum = (int )(urb->pipe >> 15) & 15;
  max3421_hcd->curr_len = 0UL;
  max3421_hcd->hien = (u8 )((unsigned int )max3421_hcd->hien | 4U);
  return (epnum);
}
}
static int max3421_transfer_out(struct usb_hcd *hcd , struct urb *urb , int fast_retransmit )
{
  struct spi_device *spi ;
  struct spi_device *tmp ;
  struct max3421_hcd *max3421_hcd ;
  struct max3421_hcd *tmp___0 ;
  int epnum ;
  u32 max_packet ;
  void *src ;
  __u16 tmp___1 ;
  u32 _min1 ;
  u32 _min2 ;
  {
  tmp = to_spi_device(hcd->self.controller);
  spi = tmp;
  tmp___0 = hcd_to_max3421(hcd);
  max3421_hcd = tmp___0;
  epnum = (int )(urb->pipe >> 15) & 15;
  src = urb->transfer_buffer + (unsigned long )urb->actual_length;
  if (fast_retransmit != 0) {
    if ((unsigned int )max3421_hcd->rev == 18U) {
      spi_wr8(hcd, 7U, 0);
      spi_wr8(hcd, 2U, (int )*((u8 *)src));
      spi_wr8(hcd, 7U, (int )((u8 )max3421_hcd->curr_len));
    } else {
    }
    return (epnum | 32);
  } else {
  }
  tmp___1 = usb_maxpacket(urb->dev, (int )urb->pipe, 1);
  max_packet = (u32 )tmp___1;
  if (max_packet > 64U) {
    dev_err((struct device const *)(& spi->dev), "%s: packet-size of %u too big (limit is %u bytes)",
            "max3421_transfer_out", max_packet, 64);
    max3421_hcd->urb_done = -90;
    return (-90);
  } else {
  }
  _min1 = urb->transfer_buffer_length - urb->actual_length;
  _min2 = max_packet;
  max3421_hcd->curr_len = (size_t )(_min1 < _min2 ? _min1 : _min2);
  spi_wr_buf(hcd, 2U, src, max3421_hcd->curr_len);
  spi_wr8(hcd, 7U, (int )((u8 )max3421_hcd->curr_len));
  return (epnum | 32);
}
}
static void max3421_next_transfer(struct usb_hcd *hcd , int fast_retransmit )
{
  struct max3421_hcd *max3421_hcd ;
  struct max3421_hcd *tmp ;
  struct urb *urb ;
  struct max3421_ep *max3421_ep ;
  int cmd ;
  int tmp___0 ;
  int tmp___1 ;
  {
  tmp = hcd_to_max3421(hcd);
  max3421_hcd = tmp;
  urb = max3421_hcd->curr_urb;
  cmd = -22;
  if ((unsigned long )urb == (unsigned long )((struct urb *)0)) {
    return;
  } else {
  }
  max3421_ep = (struct max3421_ep *)(urb->ep)->hcpriv;
  switch ((unsigned int )max3421_ep->pkt_state) {
  case 0U:
  cmd = max3421_ctrl_setup(hcd, urb);
  goto ldv_30997;
  case 1U:
  tmp___0 = usb_urb_dir_in(urb);
  if (tmp___0 != 0) {
    cmd = max3421_transfer_in(hcd, urb);
  } else {
    cmd = max3421_transfer_out(hcd, urb, fast_retransmit);
  }
  goto ldv_30997;
  case 2U:
  tmp___1 = usb_urb_dir_in(urb);
  if (tmp___1 != 0) {
    cmd = 160;
  } else {
    cmd = 128;
  }
  goto ldv_30997;
  }
  ldv_30997: ;
  if (cmd < 0) {
    return;
  } else {
  }
  spi_wr8(hcd, 30U, (int )((u8 )cmd));
  max3421_hcd->hien = (u8 )((unsigned int )max3421_hcd->hien | 128U);
  return;
}
}
static int max3421_select_and_start_urb(struct usb_hcd *hcd )
{
  struct spi_device *spi ;
  struct spi_device *tmp ;
  struct max3421_hcd *max3421_hcd ;
  struct max3421_hcd *tmp___0 ;
  struct urb *urb ;
  struct urb *curr_urb ;
  struct max3421_ep *max3421_ep ;
  int epnum ;
  int force_toggles ;
  struct usb_host_endpoint *ep ;
  struct list_head *pos ;
  unsigned long flags ;
  struct list_head const *__mptr ;
  int tmp___1 ;
  int tmp___2 ;
  struct list_head const *__mptr___0 ;
  struct _ddebug descriptor ;
  long tmp___3 ;
  int tmp___4 ;
  s16 tmp___5 ;
  s16 tmp___6 ;
  s16 tmp___7 ;
  struct urb *tmp___8 ;
  int tmp___9 ;
  {
  tmp = to_spi_device(hcd->self.controller);
  spi = tmp;
  tmp___0 = hcd_to_max3421(hcd);
  max3421_hcd = tmp___0;
  curr_urb = (struct urb *)0;
  force_toggles = 0;
  ldv_spin_lock();
  goto ldv_31035;
  ldv_31034:
  pos = max3421_hcd->ep_list.next;
  goto ldv_31032;
  ldv_31031:
  urb = (struct urb *)0;
  __mptr = (struct list_head const *)pos;
  max3421_ep = (struct max3421_ep *)__mptr + 0xfffffffffffffff8UL;
  ep = max3421_ep->ep;
  tmp___1 = usb_endpoint_type((struct usb_endpoint_descriptor const *)(& ep->desc));
  switch (tmp___1) {
  case 1: ;
  case 3: ;
  if ((unsigned int )max3421_hcd->sched_pass != 0U) {
    goto ldv_31017;
  } else {
  }
  goto ldv_31018;
  case 0: ;
  case 2: ;
  if ((unsigned int )max3421_hcd->sched_pass != 1U) {
    goto ldv_31017;
  } else {
  }
  goto ldv_31018;
  }
  ldv_31018:
  tmp___2 = list_empty((struct list_head const *)(& ep->urb_list));
  if (tmp___2 != 0) {
    goto ldv_31017;
  } else {
  }
  __mptr___0 = (struct list_head const *)ep->urb_list.next;
  urb = (struct urb *)__mptr___0 + 0xffffffffffffffe0UL;
  if (urb->unlinked != 0) {
    descriptor.modname = "max3421_hcd";
    descriptor.function = "max3421_select_and_start_urb";
    descriptor.filename = "/work/ldvuser/mutilin/launch/work/current--X--drivers--X--defaultlinux-3.16-rc1.tar.xz--X--43_2a--X--cpachecker/linux-3.16-rc1.tar.xz/csd_deg_dscv/8168/dscv_tempdir/dscv/ri/43_2a/drivers/usb/host/max3421-hcd.o.c.prepared";
    descriptor.format = "%s: URB %p unlinked=%d";
    descriptor.lineno = 747U;
    descriptor.flags = 0U;
    tmp___3 = ldv__builtin_expect((long )descriptor.flags & 1L, 0L);
    if (tmp___3 != 0L) {
      __dynamic_dev_dbg(& descriptor, (struct device const *)(& spi->dev), "%s: URB %p unlinked=%d",
                        "max3421_select_and_start_urb", urb, urb->unlinked);
    } else {
    }
    max3421_hcd->curr_urb = urb;
    max3421_hcd->urb_done = 1;
    spin_unlock_irqrestore(& max3421_hcd->lock, flags);
    return (1);
  } else {
  }
  tmp___4 = usb_endpoint_type((struct usb_endpoint_descriptor const *)(& ep->desc));
  switch (tmp___4) {
  case 0:
  tmp___5 = frame_diff((int )max3421_ep->last_active, (int )max3421_hcd->frame_number);
  if ((int )tmp___5 == 0) {
    goto ldv_31017;
  } else {
  }
  goto ldv_31026;
  case 2: ;
  if ((unsigned int )max3421_ep->retransmit != 0U) {
    tmp___6 = frame_diff((int )max3421_ep->last_active, (int )max3421_hcd->frame_number);
    if ((int )tmp___6 == 0) {
      goto ldv_31017;
    } else {
    }
  } else {
  }
  goto ldv_31026;
  case 1: ;
  case 3:
  tmp___7 = frame_diff((int )max3421_hcd->frame_number, (int )max3421_ep->last_active);
  if ((int )tmp___7 < urb->interval) {
    goto ldv_31017;
  } else {
  }
  goto ldv_31026;
  }
  ldv_31026:
  list_move_tail(pos, & max3421_hcd->ep_list);
  curr_urb = urb;
  goto done;
  ldv_31017:
  pos = pos->next;
  ldv_31032: ;
  if ((unsigned long )(& max3421_hcd->ep_list) != (unsigned long )pos) {
    goto ldv_31031;
  } else {
  }
  max3421_hcd->sched_pass = (enum scheduling_pass )((unsigned int )max3421_hcd->sched_pass + 1U);
  ldv_31035: ;
  if ((unsigned int )max3421_hcd->sched_pass <= 1U) {
    goto ldv_31034;
  } else {
  }
  done: ;
  if ((unsigned long )curr_urb == (unsigned long )((struct urb *)0)) {
    spin_unlock_irqrestore(& max3421_hcd->lock, flags);
    return (0);
  } else {
  }
  tmp___8 = curr_urb;
  max3421_hcd->curr_urb = tmp___8;
  urb = tmp___8;
  epnum = usb_endpoint_num((struct usb_endpoint_descriptor const *)(& (urb->ep)->desc));
  if ((unsigned int )max3421_ep->retransmit != 0U) {
    max3421_ep->retransmit = 0U;
  } else {
    tmp___9 = usb_endpoint_xfer_control((struct usb_endpoint_descriptor const *)(& ep->desc));
    if (tmp___9 != 0) {
      (urb->dev)->toggle[0] = ((urb->dev)->toggle[0] & (unsigned int )(~ (1 << epnum))) | (unsigned int )(1 << epnum);
      (urb->dev)->toggle[1] = ((urb->dev)->toggle[1] & (unsigned int )(~ (1 << epnum))) | (unsigned int )(1 << epnum);
      max3421_ep->pkt_state = 0;
      force_toggles = 1;
    } else {
      max3421_ep->pkt_state = 1;
    }
  }
  spin_unlock_irqrestore(& max3421_hcd->lock, flags);
  max3421_ep->last_active = max3421_hcd->frame_number;
  max3421_set_address(hcd, urb->dev, epnum, force_toggles);
  max3421_set_speed(hcd, urb->dev);
  max3421_next_transfer(hcd, 0);
  return (1);
}
}
static int max3421_check_unlink(struct usb_hcd *hcd )
{
  struct spi_device *spi ;
  struct spi_device *tmp ;
  struct max3421_hcd *max3421_hcd ;
  struct max3421_hcd *tmp___0 ;
  struct list_head *pos ;
  struct list_head *upos ;
  struct list_head *next_upos ;
  struct max3421_ep *max3421_ep ;
  struct usb_host_endpoint *ep ;
  struct urb *urb ;
  unsigned long flags ;
  int retval ;
  struct list_head const *__mptr ;
  struct list_head const *__mptr___0 ;
  struct _ddebug descriptor ;
  long tmp___1 ;
  {
  tmp = to_spi_device(hcd->self.controller);
  spi = tmp;
  tmp___0 = hcd_to_max3421(hcd);
  max3421_hcd = tmp___0;
  retval = 0;
  ldv_spin_lock();
  pos = max3421_hcd->ep_list.next;
  goto ldv_31060;
  ldv_31059:
  __mptr = (struct list_head const *)pos;
  max3421_ep = (struct max3421_ep *)__mptr + 0xfffffffffffffff8UL;
  ep = max3421_ep->ep;
  upos = ep->urb_list.next;
  next_upos = upos->next;
  goto ldv_31057;
  ldv_31056:
  __mptr___0 = (struct list_head const *)upos;
  urb = (struct urb *)__mptr___0 + 0xffffffffffffffe0UL;
  if (urb->unlinked != 0) {
    retval = 1;
    descriptor.modname = "max3421_hcd";
    descriptor.function = "max3421_check_unlink";
    descriptor.filename = "/work/ldvuser/mutilin/launch/work/current--X--drivers--X--defaultlinux-3.16-rc1.tar.xz--X--43_2a--X--cpachecker/linux-3.16-rc1.tar.xz/csd_deg_dscv/8168/dscv_tempdir/dscv/ri/43_2a/drivers/usb/host/max3421-hcd.o.c.prepared";
    descriptor.format = "%s: URB %p unlinked=%d";
    descriptor.lineno = 859U;
    descriptor.flags = 0U;
    tmp___1 = ldv__builtin_expect((long )descriptor.flags & 1L, 0L);
    if (tmp___1 != 0L) {
      __dynamic_dev_dbg(& descriptor, (struct device const *)(& spi->dev), "%s: URB %p unlinked=%d",
                        "max3421_check_unlink", urb, urb->unlinked);
    } else {
    }
    usb_hcd_unlink_urb_from_ep(hcd, urb);
    spin_unlock_irqrestore(& max3421_hcd->lock, flags);
    usb_hcd_giveback_urb(hcd, urb, 0);
    ldv_spin_lock();
  } else {
  }
  upos = next_upos;
  next_upos = upos->next;
  ldv_31057: ;
  if ((unsigned long )(& ep->urb_list) != (unsigned long )upos) {
    goto ldv_31056;
  } else {
  }
  pos = pos->next;
  ldv_31060: ;
  if ((unsigned long )(& max3421_hcd->ep_list) != (unsigned long )pos) {
    goto ldv_31059;
  } else {
  }
  spin_unlock_irqrestore(& max3421_hcd->lock, flags);
  return (retval);
}
}
static void max3421_slow_retransmit(struct usb_hcd *hcd )
{
  struct max3421_hcd *max3421_hcd ;
  struct max3421_hcd *tmp ;
  struct urb *urb ;
  struct max3421_ep *max3421_ep ;
  {
  tmp = hcd_to_max3421(hcd);
  max3421_hcd = tmp;
  urb = max3421_hcd->curr_urb;
  max3421_ep = (struct max3421_ep *)(urb->ep)->hcpriv;
  max3421_ep->retransmit = 1U;
  max3421_hcd->curr_urb = (struct urb *)0;
  return;
}
}
static void max3421_recv_data_available(struct usb_hcd *hcd )
{
  struct max3421_hcd *max3421_hcd ;
  struct max3421_hcd *tmp ;
  struct urb *urb ;
  size_t remaining ;
  size_t transfer_size ;
  u8 rcvbc ;
  void *dst ;
  {
  tmp = hcd_to_max3421(hcd);
  max3421_hcd = tmp;
  urb = max3421_hcd->curr_urb;
  rcvbc = spi_rd8(hcd, 6U);
  if ((unsigned int )rcvbc > 64U) {
    rcvbc = 64U;
  } else {
  }
  if (urb->actual_length >= urb->transfer_buffer_length) {
    remaining = 0UL;
  } else {
    remaining = (size_t )(urb->transfer_buffer_length - urb->actual_length);
  }
  transfer_size = (size_t )rcvbc;
  if (transfer_size > remaining) {
    transfer_size = remaining;
  } else {
  }
  if (transfer_size != 0UL) {
    dst = urb->transfer_buffer + (unsigned long )urb->actual_length;
    spi_rd_buf(hcd, 1U, dst, transfer_size);
    urb->actual_length = urb->actual_length + (u32 )transfer_size;
    max3421_hcd->curr_len = transfer_size;
  } else {
  }
  spi_wr8(hcd, 25U, 4);
  return;
}
}
static void max3421_handle_error(struct usb_hcd *hcd , u8 hrsl )
{
  struct spi_device *spi ;
  struct spi_device *tmp ;
  struct max3421_hcd *max3421_hcd ;
  struct max3421_hcd *tmp___0 ;
  u8 result_code ;
  struct urb *urb ;
  struct max3421_ep *max3421_ep ;
  int switch_sndfifo ;
  int tmp___1 ;
  int tmp___2 ;
  struct _ddebug descriptor ;
  long tmp___3 ;
  int sndtog ;
  int tmp___4 ;
  struct _ddebug descriptor___0 ;
  long tmp___5 ;
  u8 tmp___6 ;
  struct _ddebug descriptor___1 ;
  long tmp___7 ;
  u32 tmp___8 ;
  {
  tmp = to_spi_device(hcd->self.controller);
  spi = tmp;
  tmp___0 = hcd_to_max3421(hcd);
  max3421_hcd = tmp___0;
  result_code = (unsigned int )hrsl & 15U;
  urb = max3421_hcd->curr_urb;
  max3421_ep = (struct max3421_ep *)(urb->ep)->hcpriv;
  if ((unsigned int )max3421_ep->pkt_state == 1U) {
    tmp___1 = usb_urb_dir_out(urb);
    if (tmp___1 != 0) {
      tmp___2 = 1;
    } else {
      tmp___2 = 0;
    }
  } else {
    tmp___2 = 0;
  }
  switch_sndfifo = tmp___2;
  switch ((int )result_code) {
  case 0: ;
  return;
  case 7: ;
  case 1: ;
  case 2: ;
  case 3: ;
  case 12: ;
  case 13:
  max3421_hcd->urb_done = hrsl_to_error[(int )result_code];
  descriptor.modname = "max3421_hcd";
  descriptor.function = "max3421_handle_error";
  descriptor.filename = "/work/ldvuser/mutilin/launch/work/current--X--drivers--X--defaultlinux-3.16-rc1.tar.xz--X--43_2a--X--cpachecker/linux-3.16-rc1.tar.xz/csd_deg_dscv/8168/dscv_tempdir/dscv/ri/43_2a/drivers/usb/host/max3421-hcd.o.c.prepared";
  descriptor.format = "%s: unexpected error HRSL=0x%02x";
  descriptor.lineno = 956U;
  descriptor.flags = 0U;
  tmp___3 = ldv__builtin_expect((long )descriptor.flags & 1L, 0L);
  if (tmp___3 != 0L) {
    __dynamic_dev_dbg(& descriptor, (struct device const *)(& spi->dev), "%s: unexpected error HRSL=0x%02x",
                      "max3421_handle_error", (int )hrsl);
  } else {
  }
  goto ldv_31096;
  case 6:
  tmp___4 = usb_urb_dir_in(urb);
  if (tmp___4 != 0) {
  } else {
    sndtog = ((int )hrsl >> 5) & 1;
    sndtog = sndtog ^ 1;
    spi_wr8(hcd, 29U, (int )((u8 )(1UL << (sndtog + 6))));
  }
  case 8: ;
  case 9: ;
  case 10: ;
  case 11: ;
  case 15: ;
  case 14:
  tmp___6 = max3421_ep->retries;
  max3421_ep->retries = (u8 )((int )max3421_ep->retries + 1);
  if ((unsigned int )tmp___6 <= 2U) {
    max3421_slow_retransmit(hcd);
  } else {
    max3421_hcd->urb_done = hrsl_to_error[(int )result_code];
    descriptor___0.modname = "max3421_hcd";
    descriptor___0.function = "max3421_handle_error";
    descriptor___0.filename = "/work/ldvuser/mutilin/launch/work/current--X--drivers--X--defaultlinux-3.16-rc1.tar.xz--X--43_2a--X--cpachecker/linux-3.16-rc1.tar.xz/csd_deg_dscv/8168/dscv_tempdir/dscv/ri/43_2a/drivers/usb/host/max3421-hcd.o.c.prepared";
    descriptor___0.format = "%s: unexpected error HRSL=0x%02x";
    descriptor___0.lineno = 984U;
    descriptor___0.flags = 0U;
    tmp___5 = ldv__builtin_expect((long )descriptor___0.flags & 1L, 0L);
    if (tmp___5 != 0L) {
      __dynamic_dev_dbg(& descriptor___0, (struct device const *)(& spi->dev), "%s: unexpected error HRSL=0x%02x",
                        "max3421_handle_error", (int )hrsl);
    } else {
    }
  }
  goto ldv_31096;
  case 5:
  descriptor___1.modname = "max3421_hcd";
  descriptor___1.function = "max3421_handle_error";
  descriptor___1.filename = "/work/ldvuser/mutilin/launch/work/current--X--drivers--X--defaultlinux-3.16-rc1.tar.xz--X--43_2a--X--cpachecker/linux-3.16-rc1.tar.xz/csd_deg_dscv/8168/dscv_tempdir/dscv/ri/43_2a/drivers/usb/host/max3421-hcd.o.c.prepared";
  descriptor___1.format = "%s: unexpected error HRSL=0x%02x";
  descriptor___1.lineno = 990U;
  descriptor___1.flags = 0U;
  tmp___7 = ldv__builtin_expect((long )descriptor___1.flags & 1L, 0L);
  if (tmp___7 != 0L) {
    __dynamic_dev_dbg(& descriptor___1, (struct device const *)(& spi->dev), "%s: unexpected error HRSL=0x%02x",
                      "max3421_handle_error", (int )hrsl);
  } else {
  }
  max3421_hcd->urb_done = hrsl_to_error[(int )result_code];
  goto ldv_31096;
  case 4:
  tmp___8 = max3421_ep->naks;
  max3421_ep->naks = max3421_ep->naks + 1U;
  if (tmp___8 <= 1U) {
    max3421_next_transfer(hcd, 1);
    switch_sndfifo = 0;
  } else {
    max3421_slow_retransmit(hcd);
  }
  goto ldv_31096;
  }
  ldv_31096: ;
  if (switch_sndfifo != 0) {
    spi_wr8(hcd, 7U, 0);
  } else {
  }
  return;
}
}
static int max3421_transfer_in_done(struct usb_hcd *hcd , struct urb *urb )
{
  struct spi_device *spi ;
  struct spi_device *tmp ;
  struct max3421_hcd *max3421_hcd ;
  struct max3421_hcd *tmp___0 ;
  u32 max_packet ;
  __u16 tmp___1 ;
  {
  tmp = to_spi_device(hcd->self.controller);
  spi = tmp;
  tmp___0 = hcd_to_max3421(hcd);
  max3421_hcd = tmp___0;
  if (urb->actual_length >= urb->transfer_buffer_length) {
    return (1);
  } else {
  }
  tmp___1 = usb_maxpacket(urb->dev, (int )urb->pipe, 0);
  max_packet = (u32 )tmp___1;
  if (max_packet > 64U) {
    dev_err((struct device const *)(& spi->dev), "%s: packet-size of %u too big (limit is %u bytes)",
            "max3421_transfer_in_done", max_packet, 64);
    return (-22);
  } else {
  }
  if (max3421_hcd->curr_len < (size_t )max_packet) {
    if ((int )urb->transfer_flags & 1) {
      return (-121);
    } else {
      return (1);
    }
  } else {
  }
  return (0);
}
}
static int max3421_transfer_out_done(struct usb_hcd *hcd , struct urb *urb )
{
  struct max3421_hcd *max3421_hcd ;
  struct max3421_hcd *tmp ;
  u32 max_packet ;
  __u16 tmp___0 ;
  {
  tmp = hcd_to_max3421(hcd);
  max3421_hcd = tmp;
  urb->actual_length = urb->actual_length + (u32 )max3421_hcd->curr_len;
  if (urb->actual_length < urb->transfer_buffer_length) {
    return (0);
  } else {
  }
  if ((urb->transfer_flags & 64U) != 0U) {
    tmp___0 = usb_maxpacket(urb->dev, (int )urb->pipe, 1);
    max_packet = (u32 )tmp___0;
    if (max3421_hcd->curr_len == (size_t )max_packet) {
      return (0);
    } else {
    }
  } else {
  }
  return (1);
}
}
static void max3421_host_transfer_done(struct usb_hcd *hcd )
{
  struct max3421_hcd *max3421_hcd ;
  struct max3421_hcd *tmp ;
  struct urb *urb ;
  struct max3421_ep *max3421_ep ;
  u8 result_code ;
  u8 hrsl ;
  int urb_done ;
  long tmp___0 ;
  int tmp___1 ;
  {
  tmp = hcd_to_max3421(hcd);
  max3421_hcd = tmp;
  urb = max3421_hcd->curr_urb;
  urb_done = 0;
  max3421_hcd->hien = (unsigned int )max3421_hcd->hien & 123U;
  hrsl = spi_rd8(hcd, 31U);
  result_code = (unsigned int )hrsl & 15U;
  max3421_ep = (struct max3421_ep *)(urb->ep)->hcpriv;
  tmp___0 = ldv__builtin_expect((unsigned int )result_code != 0U, 0L);
  if (tmp___0 != 0L) {
    max3421_handle_error(hcd, (int )hrsl);
    return;
  } else {
  }
  max3421_ep->naks = 0U;
  max3421_ep->retries = 0U;
  switch ((unsigned int )max3421_ep->pkt_state) {
  case 0U: ;
  if (urb->transfer_buffer_length != 0U) {
    max3421_ep->pkt_state = 1;
  } else {
    max3421_ep->pkt_state = 2;
  }
  goto ldv_31133;
  case 1U:
  tmp___1 = usb_urb_dir_in(urb);
  if (tmp___1 != 0) {
    urb_done = max3421_transfer_in_done(hcd, urb);
  } else {
    urb_done = max3421_transfer_out_done(hcd, urb);
  }
  if (urb_done > 0 && urb->pipe >> 30 == 2U) {
    urb_done = 0;
    max3421_hcd->urb_done = urb_done;
    max3421_ep->pkt_state = 2;
  } else {
  }
  goto ldv_31133;
  case 2U:
  urb_done = 1;
  goto ldv_31133;
  }
  ldv_31133: ;
  if (urb_done != 0) {
    max3421_hcd->urb_done = urb_done;
  } else {
    max3421_next_transfer(hcd, 0);
  }
  return;
}
}
static void max3421_detect_conn(struct usb_hcd *hcd )
{
  struct max3421_hcd *max3421_hcd ;
  struct max3421_hcd *tmp ;
  unsigned int jk ;
  unsigned int have_conn ;
  u32 old_port_status ;
  u32 chg ;
  unsigned long flags ;
  u8 hrsl ;
  u8 mode ;
  {
  tmp = hcd_to_max3421(hcd);
  max3421_hcd = tmp;
  have_conn = 0U;
  hrsl = spi_rd8(hcd, 31U);
  jk = (unsigned int )((((int )hrsl >> 7) & 1) | ((((int )hrsl >> 6) & 1) << 1));
  mode = max3421_hcd->mode;
  switch (jk) {
  case 0U:
  mode = (unsigned int )mode & 247U;
  goto ldv_31148;
  case 1U: ;
  case 2U: ;
  if (jk == 2U) {
    mode = (u8 )((unsigned int )mode ^ 2U);
  } else {
  }
  mode = (u8 )((unsigned int )mode | 8U);
  have_conn = 1U;
  goto ldv_31148;
  case 3U: ;
  goto ldv_31148;
  }
  ldv_31148:
  max3421_hcd->mode = mode;
  spi_wr8(hcd, 27U, (int )max3421_hcd->mode);
  ldv_spin_lock();
  old_port_status = max3421_hcd->port_status;
  if (have_conn != 0U) {
    max3421_hcd->port_status = max3421_hcd->port_status | 1U;
  } else {
    max3421_hcd->port_status = max3421_hcd->port_status & 4294967294U;
  }
  if (((unsigned long )mode & 2UL) != 0UL) {
    max3421_hcd->port_status = max3421_hcd->port_status | 512U;
  } else {
    max3421_hcd->port_status = max3421_hcd->port_status & 4294966783U;
  }
  chg = max3421_hcd->port_status ^ old_port_status;
  max3421_hcd->port_status = max3421_hcd->port_status | (chg << 16);
  spin_unlock_irqrestore(& max3421_hcd->lock, flags);
  return;
}
}
static irqreturn_t max3421_irq_handler(int irq , void *dev_id )
{
  struct usb_hcd *hcd ;
  struct spi_device *spi ;
  struct spi_device *tmp ;
  struct max3421_hcd *max3421_hcd ;
  struct max3421_hcd *tmp___0 ;
  {
  hcd = (struct usb_hcd *)dev_id;
  tmp = to_spi_device(hcd->self.controller);
  spi = tmp;
  tmp___0 = hcd_to_max3421(hcd);
  max3421_hcd = tmp___0;
  if ((unsigned long )max3421_hcd->spi_thread != (unsigned long )((struct task_struct *)0) && (long )(max3421_hcd->spi_thread)->state != 0L) {
    wake_up_process(max3421_hcd->spi_thread);
  } else {
  }
  if ((unsigned int )*((unsigned char *)max3421_hcd + 188UL) == 0U) {
    max3421_hcd->do_enable_irq = 1U;
    disable_irq_nosync((unsigned int )spi->irq);
  } else {
  }
  return (1);
}
}
static int max3421_handle_irqs(struct usb_hcd *hcd )
{
  struct max3421_hcd *max3421_hcd ;
  struct max3421_hcd *tmp ;
  u32 chg ;
  u32 old_port_status ;
  unsigned long flags ;
  u8 hirq ;
  {
  tmp = hcd_to_max3421(hcd);
  max3421_hcd = tmp;
  hirq = spi_rd8(hcd, 25U);
  hirq = (u8 )((int )max3421_hcd->hien & (int )hirq);
  if ((unsigned int )hirq == 0U) {
    return (0);
  } else {
  }
  spi_wr8(hcd, 25U, (int )hirq & 243);
  if (((unsigned long )hirq & 64UL) != 0UL) {
    max3421_hcd->frame_number = (unsigned int )((u16 )((unsigned int )max3421_hcd->frame_number + 1U)) & 2047U;
    max3421_hcd->sched_pass = 0;
  } else {
  }
  if (((unsigned long )hirq & 4UL) != 0UL) {
    max3421_recv_data_available(hcd);
  } else {
  }
  if ((int )((signed char )hirq) < 0) {
    max3421_host_transfer_done(hcd);
  } else {
  }
  if (((unsigned long )hirq & 32UL) != 0UL) {
    max3421_detect_conn(hcd);
  } else {
  }
  ldv_spin_lock();
  old_port_status = max3421_hcd->port_status;
  if ((int )hirq & 1) {
    if ((max3421_hcd->port_status & 16U) != 0U) {
      max3421_hcd->port_status = max3421_hcd->port_status & 4294967279U;
      max3421_hcd->port_status = max3421_hcd->port_status | 2U;
    } else {
      printk("\016%s: BUSEVENT Bus Resume Done\n", "max3421_handle_irqs");
    }
  } else {
  }
  if (((unsigned long )hirq & 2UL) != 0UL) {
    printk("\016%s: RWU\n", "max3421_handle_irqs");
  } else {
  }
  if (((unsigned long )hirq & 16UL) != 0UL) {
    printk("\016%s: SUSDN\n", "max3421_handle_irqs");
  } else {
  }
  chg = max3421_hcd->port_status ^ old_port_status;
  max3421_hcd->port_status = max3421_hcd->port_status | (chg << 16);
  spin_unlock_irqrestore(& max3421_hcd->lock, flags);
  return (1);
}
}
static int max3421_reset_hcd(struct usb_hcd *hcd )
{
  struct spi_device *spi ;
  struct spi_device *tmp ;
  struct max3421_hcd *max3421_hcd ;
  struct max3421_hcd *tmp___0 ;
  int timeout ;
  u8 tmp___1 ;
  {
  tmp = to_spi_device(hcd->self.controller);
  spi = tmp;
  tmp___0 = hcd_to_max3421(hcd);
  max3421_hcd = tmp___0;
  spi_wr8(hcd, 15U, 32);
  spi_wr8(hcd, 15U, 0);
  timeout = 1000;
  ldv_31176:
  tmp___1 = spi_rd8(hcd, 13U);
  if ((int )tmp___1 & 1) {
    goto ldv_31174;
  } else {
  }
  timeout = timeout - 1;
  if (timeout < 0) {
    dev_err((struct device const *)(& spi->dev), "timed out waiting for oscillator OK signal");
    return (1);
  } else {
  }
  __might_sleep("/work/ldvuser/mutilin/launch/work/current--X--drivers--X--defaultlinux-3.16-rc1.tar.xz--X--43_2a--X--cpachecker/linux-3.16-rc1.tar.xz/csd_deg_dscv/8168/dscv_tempdir/dscv/ri/43_2a/drivers/usb/host/max3421-hcd.o.c.prepared",
                1384, 0);
  _cond_resched();
  goto ldv_31176;
  ldv_31174:
  max3421_hcd->mode = 201U;
  spi_wr8(hcd, 27U, (int )max3421_hcd->mode);
  max3421_hcd->frame_number = 2047U;
  spi_wr8(hcd, 29U, 2);
  spi_wr8(hcd, 29U, 4);
  max3421_detect_conn(hcd);
  max3421_hcd->hien = 97U;
  spi_wr8(hcd, 26U, (int )max3421_hcd->hien);
  spi_wr8(hcd, 16U, 1);
  return (1);
}
}
static int max3421_urb_done(struct usb_hcd *hcd )
{
  struct max3421_hcd *max3421_hcd ;
  struct max3421_hcd *tmp ;
  unsigned long flags ;
  struct urb *urb ;
  int status ;
  {
  tmp = hcd_to_max3421(hcd);
  max3421_hcd = tmp;
  status = max3421_hcd->urb_done;
  max3421_hcd->urb_done = 0;
  if (status > 0) {
    status = 0;
  } else {
  }
  urb = max3421_hcd->curr_urb;
  if ((unsigned long )urb != (unsigned long )((struct urb *)0)) {
    max3421_hcd->curr_urb = (struct urb *)0;
    ldv_spin_lock();
    usb_hcd_unlink_urb_from_ep(hcd, urb);
    spin_unlock_irqrestore(& max3421_hcd->lock, flags);
    usb_hcd_giveback_urb(hcd, urb, status);
  } else {
  }
  return (1);
}
}
static int max3421_spi_thread(void *dev_id )
{
  struct usb_hcd *hcd ;
  struct spi_device *spi ;
  struct spi_device *tmp ;
  struct max3421_hcd *max3421_hcd ;
  struct max3421_hcd *tmp___0 ;
  int i ;
  int i_worked ;
  bool tmp___1 ;
  int tmp___2 ;
  long volatile __ret ;
  struct task_struct *tmp___3 ;
  struct task_struct *tmp___4 ;
  struct task_struct *tmp___5 ;
  struct task_struct *tmp___6 ;
  struct task_struct *tmp___7 ;
  int tmp___8 ;
  int tmp___9 ;
  int tmp___10 ;
  int tmp___11 ;
  int tmp___12 ;
  u8 val ;
  u8 tmp___13 ;
  bool tmp___14 ;
  int tmp___15 ;
  long volatile __ret___0 ;
  struct task_struct *tmp___16 ;
  struct task_struct *tmp___17 ;
  struct task_struct *tmp___18 ;
  struct task_struct *tmp___19 ;
  {
  hcd = (struct usb_hcd *)dev_id;
  tmp = to_spi_device(hcd->self.controller);
  spi = tmp;
  tmp___0 = hcd_to_max3421(hcd);
  max3421_hcd = tmp___0;
  i_worked = 1;
  spi_wr8(hcd, 17U, 24);
  goto ldv_31194;
  ldv_31193:
  max3421_hcd->rev = spi_rd8(hcd, 18U);
  if ((unsigned int )max3421_hcd->rev == 18U || (unsigned int )max3421_hcd->rev == 19U) {
    goto ldv_31192;
  } else {
  }
  dev_err((struct device const *)(& spi->dev), "bad rev 0x%02x", (int )max3421_hcd->rev);
  msleep(10000U);
  ldv_31194:
  tmp___1 = kthread_should_stop();
  if (tmp___1) {
    tmp___2 = 0;
  } else {
    tmp___2 = 1;
  }
  if (tmp___2) {
    goto ldv_31193;
  } else {
  }
  ldv_31192:
  _dev_info((struct device const *)(& spi->dev), "rev 0x%x, SPI clk %dHz, bpw %u, irq %d\n",
            (int )max3421_hcd->rev, spi->max_speed_hz, (int )spi->bits_per_word, spi->irq);
  goto ldv_31210;
  ldv_31209: ;
  if (i_worked == 0) {
    spi_wr8(hcd, 26U, (int )max3421_hcd->hien);
    __ret = 1L;
    switch (8UL) {
    case 1UL:
    tmp___3 = get_current();
    __asm__ volatile ("xchgb %b0, %1\n": "+q" (__ret), "+m" (tmp___3->state): : "memory",
                         "cc");
    goto ldv_31197;
    case 2UL:
    tmp___4 = get_current();
    __asm__ volatile ("xchgw %w0, %1\n": "+r" (__ret), "+m" (tmp___4->state): : "memory",
                         "cc");
    goto ldv_31197;
    case 4UL:
    tmp___5 = get_current();
    __asm__ volatile ("xchgl %0, %1\n": "+r" (__ret), "+m" (tmp___5->state): : "memory",
                         "cc");
    goto ldv_31197;
    case 8UL:
    tmp___6 = get_current();
    __asm__ volatile ("xchgq %q0, %1\n": "+r" (__ret), "+m" (tmp___6->state): : "memory",
                         "cc");
    goto ldv_31197;
    default:
    __xchg_wrong_size();
    }
    ldv_31197: ;
    if ((unsigned int )*((unsigned char *)max3421_hcd + 188UL) != 0U) {
      max3421_hcd->do_enable_irq = 0U;
      enable_irq((unsigned int )spi->irq);
    } else {
    }
    schedule();
    tmp___7 = get_current();
    tmp___7->state = 0L;
  } else {
  }
  i_worked = 0;
  if (max3421_hcd->urb_done != 0) {
    tmp___8 = max3421_urb_done(hcd);
    i_worked = tmp___8 | i_worked;
  } else {
    tmp___10 = max3421_handle_irqs(hcd);
    if (tmp___10 != 0) {
      i_worked = 1;
    } else
    if ((unsigned long )max3421_hcd->curr_urb == (unsigned long )((struct urb *)0)) {
      tmp___9 = max3421_select_and_start_urb(hcd);
      i_worked = tmp___9 | i_worked;
    } else {
    }
  }
  if ((unsigned int )*((unsigned char *)max3421_hcd + 188UL) != 0U) {
    max3421_hcd->do_reset_hcd = 0U;
    tmp___11 = max3421_reset_hcd(hcd);
    i_worked = tmp___11 | i_worked;
  } else {
  }
  if ((unsigned int )*((unsigned char *)max3421_hcd + 188UL) != 0U) {
    max3421_hcd->do_reset_port = 0U;
    spi_wr8(hcd, 29U, 1);
    i_worked = 1;
  } else {
  }
  if ((unsigned int )*((unsigned char *)max3421_hcd + 188UL) != 0U) {
    max3421_hcd->do_check_unlink = 0U;
    tmp___12 = max3421_check_unlink(hcd);
    i_worked = tmp___12 | i_worked;
  } else {
  }
  if ((unsigned int )*((unsigned char *)max3421_hcd + 188UL) != 0U) {
    i = 0;
    goto ldv_31207;
    ldv_31206:
    tmp___13 = spi_rd8(hcd, 20U);
    val = tmp___13;
    val = (u8 )(((int )((signed char )val) & -16) | ((int )((signed char )max3421_hcd->iopins[i]) & 15));
    spi_wr8(hcd, (unsigned int )(i + 20), (int )val);
    max3421_hcd->iopins[i] = val;
    i = i + 1;
    ldv_31207: ;
    if ((unsigned int )i <= 1U) {
      goto ldv_31206;
    } else {
    }
    max3421_hcd->do_iopin_update = 0U;
    i_worked = 1;
  } else {
  }
  ldv_31210:
  tmp___14 = kthread_should_stop();
  if (tmp___14) {
    tmp___15 = 0;
  } else {
    tmp___15 = 1;
  }
  if (tmp___15) {
    goto ldv_31209;
  } else {
  }
  __ret___0 = 0L;
  switch (8UL) {
  case 1UL:
  tmp___16 = get_current();
  __asm__ volatile ("xchgb %b0, %1\n": "+q" (__ret___0), "+m" (tmp___16->state): : "memory",
                       "cc");
  goto ldv_31214;
  case 2UL:
  tmp___17 = get_current();
  __asm__ volatile ("xchgw %w0, %1\n": "+r" (__ret___0), "+m" (tmp___17->state): : "memory",
                       "cc");
  goto ldv_31214;
  case 4UL:
  tmp___18 = get_current();
  __asm__ volatile ("xchgl %0, %1\n": "+r" (__ret___0), "+m" (tmp___18->state): : "memory",
                       "cc");
  goto ldv_31214;
  case 8UL:
  tmp___19 = get_current();
  __asm__ volatile ("xchgq %q0, %1\n": "+r" (__ret___0), "+m" (tmp___19->state): : "memory",
                       "cc");
  goto ldv_31214;
  default:
  __xchg_wrong_size();
  }
  ldv_31214:
  _dev_info((struct device const *)(& spi->dev), "SPI thread exiting");
  return (0);
}
}
static int max3421_reset_port(struct usb_hcd *hcd )
{
  struct max3421_hcd *max3421_hcd ;
  struct max3421_hcd *tmp ;
  {
  tmp = hcd_to_max3421(hcd);
  max3421_hcd = tmp;
  max3421_hcd->port_status = max3421_hcd->port_status & 4294966781U;
  max3421_hcd->do_reset_port = 1U;
  wake_up_process(max3421_hcd->spi_thread);
  return (0);
}
}
static int max3421_reset(struct usb_hcd *hcd )
{
  struct max3421_hcd *max3421_hcd ;
  struct max3421_hcd *tmp ;
  {
  tmp = hcd_to_max3421(hcd);
  max3421_hcd = tmp;
  hcd->self.sg_tablesize = 0U;
  hcd->speed = 32;
  (hcd->self.root_hub)->speed = 2;
  max3421_hcd->do_reset_hcd = 1U;
  wake_up_process(max3421_hcd->spi_thread);
  return (0);
}
}
static int max3421_start(struct usb_hcd *hcd )
{
  struct max3421_hcd *max3421_hcd ;
  struct max3421_hcd *tmp ;
  struct lock_class_key __key ;
  {
  tmp = hcd_to_max3421(hcd);
  max3421_hcd = tmp;
  spinlock_check(& max3421_hcd->lock);
  __raw_spin_lock_init(& max3421_hcd->lock.ldv_6347.rlock, "&(&max3421_hcd->lock)->rlock",
                       & __key);
  max3421_hcd->rh_state = 2;
  INIT_LIST_HEAD(& max3421_hcd->ep_list);
  hcd->power_budget = 500U;
  hcd->state = 1;
  hcd->uses_new_polling = 1U;
  return (0);
}
}
static void max3421_stop(struct usb_hcd *hcd )
{
  {
  return;
}
}
static int max3421_urb_enqueue(struct usb_hcd *hcd , struct urb *urb , gfp_t mem_flags )
{
  struct spi_device *spi ;
  struct spi_device *tmp ;
  struct max3421_hcd *max3421_hcd ;
  struct max3421_hcd *tmp___0 ;
  struct max3421_ep *max3421_ep ;
  unsigned long flags ;
  int retval ;
  void *tmp___1 ;
  {
  tmp = to_spi_device(hcd->self.controller);
  spi = tmp;
  tmp___0 = hcd_to_max3421(hcd);
  max3421_hcd = tmp___0;
  switch (urb->pipe >> 30) {
  case 1U: ;
  case 0U: ;
  if (urb->interval < 0) {
    dev_err((struct device const *)(& spi->dev), "%s: interval=%d for intr-/iso-pipe; expected > 0\n",
            "max3421_urb_enqueue", urb->interval);
    return (-22);
  } else {
  }
  default: ;
  goto ldv_31250;
  }
  ldv_31250:
  ldv_spin_lock();
  max3421_ep = (struct max3421_ep *)(urb->ep)->hcpriv;
  if ((unsigned long )max3421_ep == (unsigned long )((struct max3421_ep *)0)) {
    tmp___1 = kzalloc(40UL, mem_flags);
    max3421_ep = (struct max3421_ep *)tmp___1;
    if ((unsigned long )max3421_ep == (unsigned long )((struct max3421_ep *)0)) {
      retval = -12;
      goto out;
    } else {
    }
    max3421_ep->ep = urb->ep;
    max3421_ep->last_active = max3421_hcd->frame_number;
    (urb->ep)->hcpriv = (void *)max3421_ep;
    list_add_tail(& max3421_ep->ep_list, & max3421_hcd->ep_list);
  } else {
  }
  retval = usb_hcd_link_urb_to_ep(hcd, urb);
  if (retval == 0) {
    max3421_hcd->sched_pass = 0;
    wake_up_process(max3421_hcd->spi_thread);
  } else {
  }
  out:
  spin_unlock_irqrestore(& max3421_hcd->lock, flags);
  return (retval);
}
}
static int max3421_urb_dequeue(struct usb_hcd *hcd , struct urb *urb , int status )
{
  struct max3421_hcd *max3421_hcd ;
  struct max3421_hcd *tmp ;
  unsigned long flags ;
  int retval ;
  {
  tmp = hcd_to_max3421(hcd);
  max3421_hcd = tmp;
  ldv_spin_lock();
  retval = usb_hcd_check_unlink_urb(hcd, urb, status);
  if (retval == 0) {
    max3421_hcd->do_check_unlink = 1U;
    wake_up_process(max3421_hcd->spi_thread);
  } else {
  }
  spin_unlock_irqrestore(& max3421_hcd->lock, flags);
  return (retval);
}
}
static void max3421_endpoint_disable(struct usb_hcd *hcd , struct usb_host_endpoint *ep )
{
  struct max3421_hcd *max3421_hcd ;
  struct max3421_hcd *tmp ;
  unsigned long flags ;
  struct max3421_ep *max3421_ep ;
  int tmp___0 ;
  {
  tmp = hcd_to_max3421(hcd);
  max3421_hcd = tmp;
  ldv_spin_lock();
  if ((unsigned long )ep->hcpriv != (unsigned long )((void *)0)) {
    max3421_ep = (struct max3421_ep *)ep->hcpriv;
    tmp___0 = list_empty((struct list_head const *)(& max3421_ep->ep_list));
    if (tmp___0 == 0) {
      list_del(& max3421_ep->ep_list);
    } else {
    }
    kfree((void const *)max3421_ep);
    ep->hcpriv = (void *)0;
  } else {
  }
  spin_unlock_irqrestore(& max3421_hcd->lock, flags);
  return;
}
}
static int max3421_get_frame_number(struct usb_hcd *hcd )
{
  struct max3421_hcd *max3421_hcd ;
  struct max3421_hcd *tmp ;
  {
  tmp = hcd_to_max3421(hcd);
  max3421_hcd = tmp;
  return ((int )max3421_hcd->frame_number);
}
}
static int max3421_hub_status_data(struct usb_hcd *hcd , char *buf )
{
  struct max3421_hcd *max3421_hcd ;
  struct max3421_hcd *tmp ;
  unsigned long flags ;
  int retval ;
  struct _ddebug descriptor ;
  long tmp___0 ;
  {
  tmp = hcd_to_max3421(hcd);
  max3421_hcd = tmp;
  retval = 0;
  ldv_spin_lock();
  if ((hcd->flags & 1UL) == 0UL) {
    goto done;
  } else {
  }
  *buf = 0;
  if ((max3421_hcd->port_status & 2031616U) != 0U) {
    *buf = 2;
    descriptor.modname = "max3421_hcd";
    descriptor.function = "max3421_hub_status_data";
    descriptor.filename = "/work/ldvuser/mutilin/launch/work/current--X--drivers--X--defaultlinux-3.16-rc1.tar.xz--X--43_2a--X--cpachecker/linux-3.16-rc1.tar.xz/csd_deg_dscv/8168/dscv_tempdir/dscv/ri/43_2a/drivers/usb/host/max3421-hcd.o.c.prepared";
    descriptor.format = "port status 0x%08x has changes\n";
    descriptor.lineno = 1697U;
    descriptor.flags = 0U;
    tmp___0 = ldv__builtin_expect((long )descriptor.flags & 1L, 0L);
    if (tmp___0 != 0L) {
      __dynamic_dev_dbg(& descriptor, (struct device const *)hcd->self.controller,
                        "port status 0x%08x has changes\n", max3421_hcd->port_status);
    } else {
    }
    retval = 1;
    if ((unsigned int )max3421_hcd->rh_state == 1U) {
      usb_hcd_resume_root_hub(hcd);
    } else {
    }
  } else {
  }
  done:
  spin_unlock_irqrestore(& max3421_hcd->lock, flags);
  return (retval);
}
}
__inline static void hub_descriptor(struct usb_hub_descriptor *desc )
{
  {
  memset((void *)desc, 0, 15UL);
  desc->bDescriptorType = 41U;
  desc->bDescLength = 9U;
  desc->wHubCharacteristics = 1U;
  desc->bNbrPorts = 1U;
  return;
}
}
static void max3421_gpout_set_value(struct usb_hcd *hcd , u8 pin_number , u8 value )
{
  struct max3421_hcd *max3421_hcd ;
  struct max3421_hcd *tmp ;
  u8 mask ;
  u8 idx ;
  {
  tmp = hcd_to_max3421(hcd);
  max3421_hcd = tmp;
  pin_number = (u8 )((int )pin_number - 1);
  if ((unsigned int )pin_number > 7U) {
    return;
  } else {
  }
  mask = (u8 )(1U << (int )pin_number);
  idx = (u8 )((unsigned int )pin_number / 4U);
  if ((unsigned int )value != 0U) {
    max3421_hcd->iopins[(int )idx] = (u8 )((int )max3421_hcd->iopins[(int )idx] | (int )mask);
  } else {
    max3421_hcd->iopins[(int )idx] = (u8 )((int )((signed char )max3421_hcd->iopins[(int )idx]) & ~ ((int )((signed char )mask)));
  }
  max3421_hcd->do_iopin_update = 1U;
  wake_up_process(max3421_hcd->spi_thread);
  return;
}
}
static int max3421_hub_control(struct usb_hcd *hcd , u16 type_req , u16 value , u16 index ,
                               char *buf , u16 length )
{
  struct spi_device *spi ;
  struct spi_device *tmp ;
  struct max3421_hcd *max3421_hcd ;
  struct max3421_hcd *tmp___0 ;
  struct max3421_hcd_platform_data *pdata ;
  unsigned long flags ;
  int retval ;
  struct _ddebug descriptor ;
  long tmp___1 ;
  struct _ddebug descriptor___0 ;
  long tmp___2 ;
  struct _ddebug descriptor___1 ;
  long tmp___3 ;
  {
  tmp = to_spi_device(hcd->self.controller);
  spi = tmp;
  tmp___0 = hcd_to_max3421(hcd);
  max3421_hcd = tmp___0;
  retval = 0;
  ldv_spin_lock();
  pdata = (struct max3421_hcd_platform_data *)spi->dev.platform_data;
  switch ((int )type_req) {
  case 8193: ;
  goto ldv_31306;
  case 8961: ;
  switch ((int )value) {
  case 2: ;
  goto ldv_31309;
  case 8:
  descriptor.modname = "max3421_hcd";
  descriptor.function = "max3421_hub_control";
  descriptor.filename = "/work/ldvuser/mutilin/launch/work/current--X--drivers--X--defaultlinux-3.16-rc1.tar.xz--X--43_2a--X--cpachecker/linux-3.16-rc1.tar.xz/csd_deg_dscv/8168/dscv_tempdir/dscv/ri/43_2a/drivers/usb/host/max3421-hcd.o.c.prepared";
  descriptor.format = "power-off\n";
  descriptor.lineno = 1768U;
  descriptor.flags = 0U;
  tmp___1 = ldv__builtin_expect((long )descriptor.flags & 1L, 0L);
  if (tmp___1 != 0L) {
    __dynamic_dev_dbg(& descriptor, (struct device const *)hcd->self.controller,
                      "power-off\n");
  } else {
  }
  max3421_gpout_set_value(hcd, (int )pdata->vbus_gpout, (unsigned int )pdata->vbus_active_level == 0U);
  default:
  max3421_hcd->port_status = max3421_hcd->port_status & (u32 )(~ (1 << (int )value));
  }
  ldv_31309: ;
  goto ldv_31306;
  case 40966:
  hub_descriptor((struct usb_hub_descriptor *)buf);
  goto ldv_31306;
  case 32774: ;
  case 32781: ;
  case 12300: ;
  goto error;
  case 40960:
  *((__le32 *)buf) = 0U;
  goto ldv_31306;
  case 41728: ;
  if ((unsigned int )index != 1U) {
    retval = -32;
    goto error;
  } else {
  }
  *((__le16 *)buf) = (unsigned short )max3421_hcd->port_status;
  *((__le16 *)buf + 1UL) = (unsigned short )(max3421_hcd->port_status >> 16);
  goto ldv_31306;
  case 8195:
  retval = -32;
  goto ldv_31306;
  case 8963: ;
  switch ((int )value) {
  case 5: ;
  case 23: ;
  case 24: ;
  case 28: ;
  goto error;
  case 2: ;
  if ((unsigned int )*((unsigned char *)max3421_hcd + 96UL) != 0U) {
    max3421_hcd->port_status = max3421_hcd->port_status | 4U;
  } else {
  }
  goto ldv_31328;
  case 8:
  descriptor___0.modname = "max3421_hcd";
  descriptor___0.function = "max3421_hub_control";
  descriptor___0.filename = "/work/ldvuser/mutilin/launch/work/current--X--drivers--X--defaultlinux-3.16-rc1.tar.xz--X--43_2a--X--cpachecker/linux-3.16-rc1.tar.xz/csd_deg_dscv/8168/dscv_tempdir/dscv/ri/43_2a/drivers/usb/host/max3421-hcd.o.c.prepared";
  descriptor___0.format = "power-on\n";
  descriptor___0.lineno = 1817U;
  descriptor___0.flags = 0U;
  tmp___2 = ldv__builtin_expect((long )descriptor___0.flags & 1L, 0L);
  if (tmp___2 != 0L) {
    __dynamic_dev_dbg(& descriptor___0, (struct device const *)hcd->self.controller,
                      "power-on\n");
  } else {
  }
  max3421_hcd->port_status = max3421_hcd->port_status | 256U;
  max3421_gpout_set_value(hcd, (int )pdata->vbus_gpout, (int )pdata->vbus_active_level);
  goto ldv_31328;
  case 4:
  max3421_reset_port(hcd);
  default: ;
  if ((max3421_hcd->port_status & 256U) != 0U) {
    max3421_hcd->port_status = max3421_hcd->port_status | (u32 )(1 << (int )value);
  } else {
  }
  }
  ldv_31328: ;
  goto ldv_31306;
  default:
  descriptor___1.modname = "max3421_hcd";
  descriptor___1.function = "max3421_hub_control";
  descriptor___1.filename = "/work/ldvuser/mutilin/launch/work/current--X--drivers--X--defaultlinux-3.16-rc1.tar.xz--X--43_2a--X--cpachecker/linux-3.16-rc1.tar.xz/csd_deg_dscv/8168/dscv_tempdir/dscv/ri/43_2a/drivers/usb/host/max3421-hcd.o.c.prepared";
  descriptor___1.format = "hub control req%04x v%04x i%04x l%d\n";
  descriptor___1.lineno = 1835U;
  descriptor___1.flags = 0U;
  tmp___3 = ldv__builtin_expect((long )descriptor___1.flags & 1L, 0L);
  if (tmp___3 != 0L) {
    __dynamic_dev_dbg(& descriptor___1, (struct device const *)hcd->self.controller,
                      "hub control req%04x v%04x i%04x l%d\n", (int )type_req, (int )value,
                      (int )index, (int )length);
  } else {
  }
  error:
  retval = -32;
  }
  ldv_31306:
  spin_unlock_irqrestore(& max3421_hcd->lock, flags);
  return (retval);
}
}
static int max3421_bus_suspend(struct usb_hcd *hcd )
{
  {
  return (-1);
}
}
static int max3421_bus_resume(struct usb_hcd *hcd )
{
  {
  return (-1);
}
}
static int max3421_map_urb_for_dma(struct usb_hcd *hcd , struct urb *urb , gfp_t mem_flags )
{
  {
  return (0);
}
}
static void max3421_unmap_urb_for_dma(struct usb_hcd *hcd , struct urb *urb )
{
  {
  return;
}
}
static struct hc_driver max3421_hcd_desc =
     {"max3421", "MAX3421 USB Host-Controller Driver", 192UL, 0, 16, & max3421_reset,
    & max3421_start, 0, 0, & max3421_stop, 0, & max3421_get_frame_number, & max3421_urb_enqueue,
    & max3421_urb_dequeue, & max3421_map_urb_for_dma, & max3421_unmap_urb_for_dma,
    & max3421_endpoint_disable, 0, & max3421_hub_status_data, & max3421_hub_control,
    & max3421_bus_suspend, & max3421_bus_resume, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int max3421_probe(struct spi_device *spi )
{
  struct max3421_hcd *max3421_hcd ;
  struct usb_hcd *hcd ;
  int retval ;
  int tmp ;
  char const *tmp___0 ;
  void *tmp___1 ;
  void *tmp___2 ;
  struct task_struct *__k ;
  struct task_struct *tmp___3 ;
  bool tmp___4 ;
  int tmp___5 ;
  void *tmp___6 ;
  {
  hcd = (struct usb_hcd *)0;
  retval = -12;
  tmp = spi_setup(spi);
  if (tmp < 0) {
    dev_err((struct device const *)(& spi->dev), "Unable to setup SPI bus");
    return (-14);
  } else {
  }
  tmp___0 = dev_name((struct device const *)(& spi->dev));
  hcd = usb_create_hcd((struct hc_driver const *)(& max3421_hcd_desc), & spi->dev,
                       tmp___0);
  if ((unsigned long )hcd == (unsigned long )((struct usb_hcd *)0)) {
    dev_err((struct device const *)(& spi->dev), "failed to create HCD structure\n");
    goto error;
  } else {
  }
  set_bit(2L, (unsigned long volatile *)(& hcd->flags));
  max3421_hcd = hcd_to_max3421(hcd);
  max3421_hcd->next = max3421_hcd_list;
  max3421_hcd_list = max3421_hcd;
  INIT_LIST_HEAD(& max3421_hcd->ep_list);
  tmp___1 = kmalloc(2UL, 208U);
  max3421_hcd->tx = (struct max3421_dma_buf *)tmp___1;
  if ((unsigned long )max3421_hcd->tx == (unsigned long )((struct max3421_dma_buf *)0)) {
    dev_err((struct device const *)(& spi->dev), "failed to kmalloc tx buffer\n");
    goto error;
  } else {
  }
  tmp___2 = kmalloc(2UL, 208U);
  max3421_hcd->rx = (struct max3421_dma_buf *)tmp___2;
  if ((unsigned long )max3421_hcd->rx == (unsigned long )((struct max3421_dma_buf *)0)) {
    dev_err((struct device const *)(& spi->dev), "failed to kmalloc rx buffer\n");
    goto error;
  } else {
  }
  tmp___3 = kthread_create_on_node(& max3421_spi_thread, (void *)hcd, -1, "max3421_spi_thread");
  __k = tmp___3;
  tmp___4 = IS_ERR((void const *)__k);
  if (tmp___4) {
    tmp___5 = 0;
  } else {
    tmp___5 = 1;
  }
  if (tmp___5) {
    wake_up_process(__k);
  } else {
  }
  max3421_hcd->spi_thread = __k;
  tmp___6 = ERR_PTR(-12L);
  if ((unsigned long )((void *)max3421_hcd->spi_thread) == (unsigned long )tmp___6) {
    dev_err((struct device const *)(& spi->dev), "failed to create SPI thread (out of memory)\n");
    goto error;
  } else {
  }
  retval = usb_add_hcd(hcd, 0U, 0UL);
  if (retval != 0) {
    dev_err((struct device const *)(& spi->dev), "failed to add HCD\n");
    goto error;
  } else {
  }
  retval = ldv_request_irq_19((unsigned int )spi->irq, & max3421_irq_handler, 8UL,
                              "max3421", (void *)hcd);
  if (retval < 0) {
    dev_err((struct device const *)(& spi->dev), "failed to request irq %d\n", spi->irq);
    goto error;
  } else {
  }
  return (0);
  error: ;
  if ((unsigned long )hcd != (unsigned long )((struct usb_hcd *)0)) {
    kfree((void const *)max3421_hcd->tx);
    kfree((void const *)max3421_hcd->rx);
    if ((unsigned long )max3421_hcd->spi_thread != (unsigned long )((struct task_struct *)0)) {
      kthread_stop(max3421_hcd->spi_thread);
    } else {
    }
    usb_put_hcd(hcd);
  } else {
  }
  return (retval);
}
}
static int max3421_remove(struct spi_device *spi )
{
  struct max3421_hcd *max3421_hcd ;
  struct max3421_hcd **prev ;
  struct usb_hcd *hcd ;
  unsigned long flags ;
  {
  max3421_hcd = (struct max3421_hcd *)0;
  hcd = (struct usb_hcd *)0;
  prev = & max3421_hcd_list;
  goto ldv_31369;
  ldv_31368:
  max3421_hcd = *prev;
  hcd = max3421_to_hcd(max3421_hcd);
  if ((unsigned long )hcd->self.controller == (unsigned long )(& spi->dev)) {
    goto ldv_31367;
  } else {
  }
  prev = & (*prev)->next;
  ldv_31369: ;
  if ((unsigned long )*prev != (unsigned long )((struct max3421_hcd *)0)) {
    goto ldv_31368;
  } else {
  }
  ldv_31367: ;
  if ((unsigned long )max3421_hcd == (unsigned long )((struct max3421_hcd *)0)) {
    dev_err((struct device const *)(& spi->dev), "no MAX3421 HCD found for SPI device %p\n",
            spi);
    return (-19);
  } else {
  }
  usb_remove_hcd(hcd);
  ldv_spin_lock();
  kthread_stop(max3421_hcd->spi_thread);
  *prev = max3421_hcd->next;
  spin_unlock_irqrestore(& max3421_hcd->lock, flags);
  ldv_free_irq_20((unsigned int )spi->irq, (void *)hcd);
  usb_put_hcd(hcd);
  return (0);
}
}
static struct spi_driver max3421_driver = {0, & max3421_probe, & max3421_remove, 0, 0, 0, {"max3421-hcd", 0, & __this_module,
                                                    0, (_Bool)0, 0, 0, 0, 0, 0, 0,
                                                    0, 0, 0, 0}};
static int max3421_driver_init(void)
{
  int tmp ;
  {
  tmp = spi_register_driver(& max3421_driver);
  return (tmp);
}
}
static void max3421_driver_exit(void)
{
  {
  spi_unregister_driver(& max3421_driver);
  return;
}
}
int ldv_retval_0 ;
int ldv_retval_1 ;
extern void ldv_initialize(void) ;
extern void ldv_check_final_state(void) ;
int ldv_retval_2 ;
int ldv_irq_1(int state , int line , void *data )
{
  irqreturn_t irq_retval ;
  int tmp ;
  {
  if (state != 0) {
    tmp = __VERIFIER_nondet_int();
    switch (tmp) {
    case 0: ;
    if (state == 1) {
      LDV_IN_INTERRUPT = 2;
      irq_retval = max3421_irq_handler(line, data);
      LDV_IN_INTERRUPT = 1;
      return (state);
    } else {
    }
    goto ldv_31404;
    default:
    ldv_stop();
    }
    ldv_31404: ;
  } else {
  }
  return (state);
}
}
void activate_suitable_irq_1(int line , void *data )
{
  {
  if (ldv_irq_1_0 == 0) {
    ldv_irq_line_1_0 = line;
    ldv_irq_data_1_0 = data;
    ldv_irq_1_0 = 1;
    return;
  } else {
  }
  if (ldv_irq_1_1 == 0) {
    ldv_irq_line_1_1 = line;
    ldv_irq_data_1_1 = data;
    ldv_irq_1_1 = 1;
    return;
  } else {
  }
  if (ldv_irq_1_2 == 0) {
    ldv_irq_line_1_2 = line;
    ldv_irq_data_1_2 = data;
    ldv_irq_1_2 = 1;
    return;
  } else {
  }
  if (ldv_irq_1_3 == 0) {
    ldv_irq_line_1_3 = line;
    ldv_irq_data_1_3 = data;
    ldv_irq_1_3 = 1;
    return;
  } else {
  }
  return;
}
}
int reg_check_1(irqreturn_t (*handler)(int , void * ) )
{
  {
  if ((unsigned long )handler == (unsigned long )(& max3421_irq_handler)) {
    return (1);
  } else {
  }
  return (0);
}
}
void ldv_initialize_hc_driver_3(void)
{
  void *tmp ;
  void *tmp___0 ;
  {
  tmp = ldv_zalloc(192UL);
  max3421_hcd_desc_group0 = (struct urb *)tmp;
  tmp___0 = ldv_zalloc(968UL);
  max3421_hcd_desc_group1 = (struct usb_hcd *)tmp___0;
  return;
}
}
void choose_interrupt_1(void)
{
  int tmp ;
  {
  tmp = __VERIFIER_nondet_int();
  switch (tmp) {
  case 0:
  ldv_irq_1_0 = ldv_irq_1(ldv_irq_1_0, ldv_irq_line_1_0, ldv_irq_data_1_0);
  goto ldv_31424;
  case 1:
  ldv_irq_1_0 = ldv_irq_1(ldv_irq_1_1, ldv_irq_line_1_1, ldv_irq_data_1_1);
  goto ldv_31424;
  case 2:
  ldv_irq_1_0 = ldv_irq_1(ldv_irq_1_2, ldv_irq_line_1_2, ldv_irq_data_1_2);
  goto ldv_31424;
  case 3:
  ldv_irq_1_0 = ldv_irq_1(ldv_irq_1_3, ldv_irq_line_1_3, ldv_irq_data_1_3);
  goto ldv_31424;
  default:
  ldv_stop();
  }
  ldv_31424: ;
  return;
}
}
void ldv_initialize_spi_driver_2(void)
{
  void *tmp ;
  {
  tmp = ldv_zalloc(1496UL);
  max3421_driver_group0 = (struct spi_device *)tmp;
  return;
}
}
void disable_suitable_irq_1(int line , void *data )
{
  {
  if (ldv_irq_1_0 != 0 && line == ldv_irq_line_1_0) {
    ldv_irq_1_0 = 0;
    return;
  } else {
  }
  if (ldv_irq_1_1 != 0 && line == ldv_irq_line_1_1) {
    ldv_irq_1_1 = 0;
    return;
  } else {
  }
  if (ldv_irq_1_2 != 0 && line == ldv_irq_line_1_2) {
    ldv_irq_1_2 = 0;
    return;
  } else {
  }
  if (ldv_irq_1_3 != 0 && line == ldv_irq_line_1_3) {
    ldv_irq_1_3 = 0;
    return;
  } else {
  }
  return;
}
}
int main(void)
{
  char *ldvarg7 ;
  void *tmp ;
  u16 ldvarg3 ;
  u16 tmp___0 ;
  u16 ldvarg0 ;
  u16 tmp___1 ;
  gfp_t ldvarg5 ;
  gfp_t ldvarg6 ;
  struct usb_host_endpoint *ldvarg8 ;
  void *tmp___2 ;
  u16 ldvarg1 ;
  u16 tmp___3 ;
  char *ldvarg4 ;
  void *tmp___4 ;
  int ldvarg9 ;
  int tmp___5 ;
  u16 ldvarg2 ;
  u16 tmp___6 ;
  int tmp___7 ;
  int tmp___8 ;
  int tmp___9 ;
  int tmp___10 ;
  {
  tmp = ldv_zalloc(1UL);
  ldvarg7 = (char *)tmp;
  tmp___0 = __VERIFIER_nondet_u16();
  ldvarg3 = tmp___0;
  tmp___1 = __VERIFIER_nondet_u16();
  ldvarg0 = tmp___1;
  tmp___2 = ldv_zalloc(72UL);
  ldvarg8 = (struct usb_host_endpoint *)tmp___2;
  tmp___3 = __VERIFIER_nondet_u16();
  ldvarg1 = tmp___3;
  tmp___4 = ldv_zalloc(1UL);
  ldvarg4 = (char *)tmp___4;
  tmp___5 = __VERIFIER_nondet_int();
  ldvarg9 = tmp___5;
  tmp___6 = __VERIFIER_nondet_u16();
  ldvarg2 = tmp___6;
  ldv_initialize();
  ldvarg5 = (gfp_t)__VERIFIER_nondet_int();
  ldvarg6 = (gfp_t)__VERIFIER_nondet_int();
  ldv_state_variable_1 = 1;
  ref_cnt = 0;
  ldv_state_variable_0 = 1;
  ldv_state_variable_3 = 0;
  ldv_state_variable_2 = 0;
  ldv_31485:
  tmp___7 = __VERIFIER_nondet_int();
  switch (tmp___7) {
  case 0: ;
  if (ldv_state_variable_1 != 0) {
    choose_interrupt_1();
  } else {
  }
  goto ldv_31456;
  case 1: ;
  if (ldv_state_variable_0 != 0) {
    tmp___8 = __VERIFIER_nondet_int();
    switch (tmp___8) {
    case 0: ;
    if (ldv_state_variable_0 == 3 && ref_cnt == 0) {
      max3421_driver_exit();
      ldv_state_variable_0 = 2;
      goto ldv_final;
    } else {
    }
    goto ldv_31460;
    case 1: ;
    if (ldv_state_variable_0 == 1) {
      ldv_retval_0 = max3421_driver_init();
      if (ldv_retval_0 == 0) {
        ldv_state_variable_0 = 3;
        ldv_state_variable_2 = 1;
        ldv_initialize_spi_driver_2();
        ldv_state_variable_3 = 1;
        ldv_initialize_hc_driver_3();
      } else {
      }
      if (ldv_retval_0 != 0) {
        ldv_state_variable_0 = 2;
        goto ldv_final;
      } else {
      }
    } else {
    }
    goto ldv_31460;
    default:
    ldv_stop();
    }
    ldv_31460: ;
  } else {
  }
  goto ldv_31456;
  case 2: ;
  if (ldv_state_variable_3 != 0) {
    tmp___9 = __VERIFIER_nondet_int();
    switch (tmp___9) {
    case 0: ;
    if (ldv_state_variable_3 == 1) {
      ldv_retval_1 = max3421_start(max3421_hcd_desc_group1);
      if (ldv_retval_1 == 0) {
        ldv_state_variable_3 = 2;
        ref_cnt = ref_cnt + 1;
      } else {
      }
    } else {
    }
    goto ldv_31465;
    case 1: ;
    if (ldv_state_variable_3 == 1) {
      max3421_urb_dequeue(max3421_hcd_desc_group1, max3421_hcd_desc_group0, ldvarg9);
      ldv_state_variable_3 = 1;
    } else {
    }
    if (ldv_state_variable_3 == 2) {
      max3421_urb_dequeue(max3421_hcd_desc_group1, max3421_hcd_desc_group0, ldvarg9);
      ldv_state_variable_3 = 2;
    } else {
    }
    goto ldv_31465;
    case 2: ;
    if (ldv_state_variable_3 == 1) {
      max3421_endpoint_disable(max3421_hcd_desc_group1, ldvarg8);
      ldv_state_variable_3 = 1;
    } else {
    }
    if (ldv_state_variable_3 == 2) {
      max3421_endpoint_disable(max3421_hcd_desc_group1, ldvarg8);
      ldv_state_variable_3 = 2;
    } else {
    }
    goto ldv_31465;
    case 3: ;
    if (ldv_state_variable_3 == 1) {
      max3421_bus_resume(max3421_hcd_desc_group1);
      ldv_state_variable_3 = 1;
    } else {
    }
    if (ldv_state_variable_3 == 2) {
      max3421_bus_resume(max3421_hcd_desc_group1);
      ldv_state_variable_3 = 2;
    } else {
    }
    goto ldv_31465;
    case 4: ;
    if (ldv_state_variable_3 == 1) {
      max3421_bus_suspend(max3421_hcd_desc_group1);
      ldv_state_variable_3 = 1;
    } else {
    }
    if (ldv_state_variable_3 == 2) {
      max3421_bus_suspend(max3421_hcd_desc_group1);
      ldv_state_variable_3 = 2;
    } else {
    }
    goto ldv_31465;
    case 5: ;
    if (ldv_state_variable_3 == 1) {
      max3421_reset(max3421_hcd_desc_group1);
      ldv_state_variable_3 = 1;
    } else {
    }
    if (ldv_state_variable_3 == 2) {
      max3421_reset(max3421_hcd_desc_group1);
      ldv_state_variable_3 = 2;
    } else {
    }
    goto ldv_31465;
    case 6: ;
    if (ldv_state_variable_3 == 1) {
      max3421_hub_status_data(max3421_hcd_desc_group1, ldvarg7);
      ldv_state_variable_3 = 1;
    } else {
    }
    if (ldv_state_variable_3 == 2) {
      max3421_hub_status_data(max3421_hcd_desc_group1, ldvarg7);
      ldv_state_variable_3 = 2;
    } else {
    }
    goto ldv_31465;
    case 7: ;
    if (ldv_state_variable_3 == 1) {
      max3421_unmap_urb_for_dma(max3421_hcd_desc_group1, max3421_hcd_desc_group0);
      ldv_state_variable_3 = 1;
    } else {
    }
    if (ldv_state_variable_3 == 2) {
      max3421_unmap_urb_for_dma(max3421_hcd_desc_group1, max3421_hcd_desc_group0);
      ldv_state_variable_3 = 2;
    } else {
    }
    goto ldv_31465;
    case 8: ;
    if (ldv_state_variable_3 == 2) {
      max3421_stop(max3421_hcd_desc_group1);
      ldv_state_variable_3 = 1;
      ref_cnt = ref_cnt - 1;
    } else {
    }
    goto ldv_31465;
    case 9: ;
    if (ldv_state_variable_3 == 1) {
      max3421_map_urb_for_dma(max3421_hcd_desc_group1, max3421_hcd_desc_group0, ldvarg6);
      ldv_state_variable_3 = 1;
    } else {
    }
    if (ldv_state_variable_3 == 2) {
      max3421_map_urb_for_dma(max3421_hcd_desc_group1, max3421_hcd_desc_group0, ldvarg6);
      ldv_state_variable_3 = 2;
    } else {
    }
    goto ldv_31465;
    case 10: ;
    if (ldv_state_variable_3 == 1) {
      max3421_urb_enqueue(max3421_hcd_desc_group1, max3421_hcd_desc_group0, ldvarg5);
      ldv_state_variable_3 = 1;
    } else {
    }
    if (ldv_state_variable_3 == 2) {
      max3421_urb_enqueue(max3421_hcd_desc_group1, max3421_hcd_desc_group0, ldvarg5);
      ldv_state_variable_3 = 2;
    } else {
    }
    goto ldv_31465;
    case 11: ;
    if (ldv_state_variable_3 == 1) {
      max3421_hub_control(max3421_hcd_desc_group1, (int )ldvarg3, (int )ldvarg2, (int )ldvarg1,
                          ldvarg4, (int )ldvarg0);
      ldv_state_variable_3 = 1;
    } else {
    }
    if (ldv_state_variable_3 == 2) {
      max3421_hub_control(max3421_hcd_desc_group1, (int )ldvarg3, (int )ldvarg2, (int )ldvarg1,
                          ldvarg4, (int )ldvarg0);
      ldv_state_variable_3 = 2;
    } else {
    }
    goto ldv_31465;
    case 12: ;
    if (ldv_state_variable_3 == 1) {
      max3421_get_frame_number(max3421_hcd_desc_group1);
      ldv_state_variable_3 = 1;
    } else {
    }
    if (ldv_state_variable_3 == 2) {
      max3421_get_frame_number(max3421_hcd_desc_group1);
      ldv_state_variable_3 = 2;
    } else {
    }
    goto ldv_31465;
    default:
    ldv_stop();
    }
    ldv_31465: ;
  } else {
  }
  goto ldv_31456;
  case 3: ;
  if (ldv_state_variable_2 != 0) {
    tmp___10 = __VERIFIER_nondet_int();
    switch (tmp___10) {
    case 0: ;
    if (ldv_state_variable_2 == 1) {
      ldv_retval_2 = max3421_probe(max3421_driver_group0);
      if (ldv_retval_2 == 0) {
        ldv_state_variable_2 = 2;
        ref_cnt = ref_cnt + 1;
      } else {
      }
    } else {
    }
    goto ldv_31481;
    case 1: ;
    if (ldv_state_variable_2 == 2) {
      max3421_remove(max3421_driver_group0);
      ldv_state_variable_2 = 1;
      ref_cnt = ref_cnt - 1;
    } else {
    }
    goto ldv_31481;
    default:
    ldv_stop();
    }
    ldv_31481: ;
  } else {
  }
  goto ldv_31456;
  default:
  ldv_stop();
  }
  ldv_31456: ;
  goto ldv_31485;
  ldv_final:
  ldv_check_final_state();
  return 0;
}
}
__inline static void spin_unlock_irqrestore(spinlock_t *lock , unsigned long flags )
{
  {
  ldv_spin_unlock();
  ldv_spin_unlock_irqrestore_8(lock, flags);
  return;
}
}
void *ldv_malloc(size_t size ) ;
__inline static void *kmalloc(size_t size , gfp_t flags )
{
  {
  ldv_check_alloc_flags(flags);
  ldv_kmalloc_12(size, flags);
  return ((void *)0);
}
}
void *ldv_zalloc(size_t size ) ;
__inline static void *kzalloc(size_t size , gfp_t flags )
{
  {
  ldv_check_alloc_flags(flags);
  return ((void *)0);
}
}
__inline static int ldv_request_irq_19(unsigned int irq , irqreturn_t (*handler)(int ,
                                                                                 void * ) ,
                                       unsigned long flags , char const *name ,
                                       void *dev )
{
  ldv_func_ret_type___2 ldv_func_res ;
  int tmp ;
  int tmp___0 ;
  {
  tmp = request_irq(irq, handler, flags, name, dev);
  ldv_func_res = tmp;
  tmp___0 = reg_check_1(handler);
  if (tmp___0 != 0 && ldv_func_res == 0) {
    activate_suitable_irq_1((int )irq, dev);
  } else {
  }
  return (ldv_func_res);
}
}
void ldv_free_irq_20(unsigned int ldv_func_arg1 , void *ldv_func_arg2 )
{
  {
  free_irq(ldv_func_arg1, ldv_func_arg2);
  disable_suitable_irq_1((int )ldv_func_arg1, ldv_func_arg2);
  return;
}
}
__inline static void ldv_error(void);
int ldv_spin = 0;
void ldv_check_alloc_flags(gfp_t flags )
{
  {
  if (ldv_spin == 0 || ! (flags & 16U)) {
  } else {
    ldv_error();
  }
  return;
}
}
extern struct page___0 *ldv_some_page(void) ;
struct page___0 *ldv_check_alloc_flags_and_return_some_page(gfp_t flags )
{
  struct page___0 *tmp ;
  {
  if (ldv_spin == 0 || ! (flags & 16U)) {
  } else {
    ldv_error();
  }
  tmp = ldv_some_page();
  return (tmp);
}
}
void ldv_check_alloc_nonatomic(void)
{
  {
  if (ldv_spin == 0) {
  } else {
    ldv_error();
  }
  return;
}
}
void ldv_spin_lock(void)
{
  {
  ldv_spin = 1;
  return;
}
}
void ldv_spin_unlock(void)
{
  {
  ldv_spin = 0;
  return;
}
}
int ldv_spin_trylock(void)
{
  int is_lock ;
  {
  is_lock = ldv_undef_int();
  if (is_lock) {
    return (0);
  } else {
    ldv_spin = 1;
    return (1);
  }
}
}
int __VERIFIER_nondet_int(void);
int __dynamic_dev_dbg(struct _ddebug *arg0, const struct device *arg1, const char *arg2, ...) {
  return __VERIFIER_nondet_int();
}
void __list_add(struct list_head *arg0, struct list_head *arg1, struct list_head *arg2) {
  return;
}
void __list_del_entry(struct list_head *arg0) {
  return;
}
void __might_sleep(const char *arg0, int arg1, int arg2) {
  return;
}
void __raw_spin_lock_init(raw_spinlock_t *arg0, const char *arg1, struct lock_class_key *arg2) {
  return;
}
int __VERIFIER_nondet_int(void);
int _cond_resched() {
  return __VERIFIER_nondet_int();
}
int __VERIFIER_nondet_int(void);
int _dev_info(const struct device *arg0, const char *arg1, ...) {
  return __VERIFIER_nondet_int();
}
void _raw_spin_unlock_irqrestore(raw_spinlock_t *arg0, unsigned long arg1) {
  return;
}
int __VERIFIER_nondet_int(void);
int dev_err(const struct device *arg0, const char *arg1, ...) {
  return __VERIFIER_nondet_int();
}
void disable_irq_nosync(unsigned int arg0) {
  return;
}
void driver_unregister(struct device_driver *arg0) {
  return;
}
void enable_irq(unsigned int arg0) {
  return;
}
void free_irq(unsigned int arg0, void *arg1) {
  return;
}
void *kmem_cache_alloc(struct kmem_cache *arg0, gfp_t arg1) {
  return ldv_malloc(0UL);
}
struct task_struct *kthread_create_on_node(int (*arg0)(void *), void *arg1, int arg2, const char *arg3, ...) {
  return ldv_malloc(sizeof(struct task_struct));
}
bool __VERIFIER_nondet_bool(void);
bool kthread_should_stop() {
  return __VERIFIER_nondet_bool();
}
int __VERIFIER_nondet_int(void);
int kthread_stop(struct task_struct *arg0) {
  return __VERIFIER_nondet_int();
}
void ldv_check_final_state() {
  return;
}
void ldv_initialize() {
  return;
}
struct page___0 *ldv_some_page() {
  return ldv_malloc(sizeof(struct page___0));
}
void list_del(struct list_head *arg0) {
  return;
}
void msleep(unsigned int arg0) {
  return;
}
int __VERIFIER_nondet_int(void);
int printk(const char *arg0, ...) {
  return __VERIFIER_nondet_int();
}
int __VERIFIER_nondet_int(void);
int request_threaded_irq(unsigned int arg0, irqreturn_t (*arg1)(int, void *), irqreturn_t (*arg2)(int, void *), unsigned long arg3, const char *arg4, void *arg5) {
  return __VERIFIER_nondet_int();
}
void schedule() {
  return;
}
int __VERIFIER_nondet_int(void);
int spi_register_driver(struct spi_driver *arg0) {
  return __VERIFIER_nondet_int();
}
int __VERIFIER_nondet_int(void);
int spi_setup(struct spi_device *arg0) {
  return __VERIFIER_nondet_int();
}
int __VERIFIER_nondet_int(void);
int spi_sync(struct spi_device *arg0, struct spi_message *arg1) {
  return __VERIFIER_nondet_int();
}
int __VERIFIER_nondet_int(void);
int usb_add_hcd(struct usb_hcd *arg0, unsigned int arg1, unsigned long arg2) {
  return __VERIFIER_nondet_int();
}
struct usb_hcd *usb_create_hcd(const struct hc_driver *arg0, struct device *arg1, const char *arg2) {
  return ldv_malloc(sizeof(struct usb_hcd));
}
int __VERIFIER_nondet_int(void);
int usb_hcd_check_unlink_urb(struct usb_hcd *arg0, struct urb *arg1, int arg2) {
  return __VERIFIER_nondet_int();
}
void usb_hcd_giveback_urb(struct usb_hcd *arg0, struct urb *arg1, int arg2) {
  return;
}
int __VERIFIER_nondet_int(void);
int usb_hcd_link_urb_to_ep(struct usb_hcd *arg0, struct urb *arg1) {
  return __VERIFIER_nondet_int();
}
void usb_hcd_resume_root_hub(struct usb_hcd *arg0) {
  return;
}
void usb_hcd_unlink_urb_from_ep(struct usb_hcd *arg0, struct urb *arg1) {
  return;
}
void usb_put_hcd(struct usb_hcd *arg0) {
  return;
}
void usb_remove_hcd(struct usb_hcd *arg0) {
  return;
}
int __VERIFIER_nondet_int(void);
int wake_up_process(struct task_struct *arg0) {
  return __VERIFIER_nondet_int();
}
void warn_slowpath_null(const char *arg0, const int arg1) {
  return;
}
void free(void *);
void kfree(void const *p) {
  free((void *)p);
}
