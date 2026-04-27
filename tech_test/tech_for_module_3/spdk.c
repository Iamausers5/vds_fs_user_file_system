/**
 * spdk_nvme_test.c
 *
 * SPDK 方式：用户态轮询模式（Poll-Mode Driver）对 NVMe SSD 进行随机读写性能测试
 * 完全绕过 Linux 内核 I/O 路径，CPU sys% ≈ 0，IOPS 可达设备物理上限。
 *
 * 依赖:
 *   - SPDK（已编译并安装，或源码树内直接引用）
 *   - DPDK（SPDK 自带或系统已安装）
 *   - 大页内存已配置（通常 2 MiB 大页 ≥ 512 页）
 *   - NVMe 设备已通过 vfio-pci / uio_pci_generic 解绑内核驱动
 *
 * 编译示例（在 SPDK 源码树根目录下）:
 *   SPDK_DIR=/path/to/spdk
 *   gcc -O2 -o spdk_nvme_test spdk_nvme_test.c \
 *       -I${SPDK_DIR}/include \
 *       -L${SPDK_DIR}/build/lib \
 *       -Wl,--whole-archive \
 *       -lspdk_nvme -lspdk_env_dpdk -lspdk_util -lspdk_log \
 *       -Wl,--no-whole-archive \
 *       $(${SPDK_DIR}/scripts/pkgdep.sh --print-ldflags 2>/dev/null) \
 *       -lpthread -lnuma -ldl -lrt
 *
 * 运行示例:
 *   sudo ./spdk_nvme_test --traddr 0000:01:00.0 --rw randread  --bs 4096 --iodepth 128 --time 10
 *   sudo ./spdk_nvme_test --traddr 0000:01:00.0 --rw randwrite --bs 4096 --iodepth 128 --time 10
 *
 * 观察指标:
 *   - IOPS        : 极高，逼近 NVMe 设备物理极限
 *   - CPU sys%    : 近似 0（所有操作在用户态完成，无系统调用）
 *   - CPU user%   : 会有一定占用（轮询核心满跑）
 *   - Latency avg : 极低（微秒级）
 *
 * 注意:
 *   SPDK 的 PMD 核心会独占 CPU（忙轮询），这是换取极低延迟 / 极高 IOPS 的代价。
 *   如需精细控制核心亲和性，请在 --lcores 参数或环境变量中指定。
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>
#include <pthread.h>
#include <errno.h>

#include "spdk/stdinc.h"
#include "spdk/nvme.h"
#include "spdk/env.h"
#include "spdk/string.h"
#include "spdk/log.h"

/* ───────────────────────────── 默认参数 ───────────────────────────── */
#define DEFAULT_BS        4096
#define DEFAULT_IODEPTH   128
#define DEFAULT_TIME_SEC  10
#define MAX_IODEPTH       1024
#define MAX_NAMESPACES    16

/* ─────────────────────────── 全局控制标志 ─────────────────────────── */
static volatile int g_running     = 1;
static volatile int g_warmup_done = 0;   /* 预热阶段不计入统计 */

/* ─────────────────────────── 命令行参数 ──────────────────────────── */
static struct {
    char     traddr[64];    /* PCI BDF，如 0000:01:00.0 */
    int      is_write;
    uint32_t bs;
    int      iodepth;
    int      time_sec;
    int      warmup_sec;
} g_args = {
    .traddr     = "",
    .is_write   = 0,
    .bs         = DEFAULT_BS,
    .iodepth    = DEFAULT_IODEPTH,
    .time_sec   = DEFAULT_TIME_SEC,
    .warmup_sec = 2,
};

/* ─────────────────────────── 统计数据 ───────────────────────────── */
typedef struct {
    uint64_t ios_completed;
    uint64_t total_latency_ns;
    uint64_t min_latency_ns;
    uint64_t max_latency_ns;
    uint64_t errors;
} ns_stats_t;

/* ─────────────────────── 单个 I/O 请求上下文 ────────────────────── */
typedef struct io_task {
    void          *buf;          /* DMA 缓冲区（从 SPDK huge-page 分配）*/
    uint64_t       submit_tsc;   /* 提交时的 TSC 计数                   */
    struct ns_ctx *ns_ctx;       /* 所属命名空间上下文                  */
} io_task_t;

/* ─────────────────────── 命名空间上下文 ─────────────────────────── */
typedef struct ns_ctx {
    struct spdk_nvme_ns   *ns;
    struct spdk_nvme_qpair *qpair;
    ns_stats_t             stats;
    io_task_t              tasks[MAX_IODEPTH];
    int                    in_flight;      /* 当前 in-flight 数量 */
    uint64_t               ns_size_bytes;
    uint64_t               ns_size_blocks; /* 以设备 lba_size 为单位 */
    uint32_t               lba_size;
} ns_ctx_t;

/* ─────────────────────── 控制器上下文 ───────────────────────────── */
typedef struct ctrlr_ctx {
    struct spdk_nvme_ctrlr *ctrlr;
    ns_ctx_t               *ns_ctxs[MAX_NAMESPACES];
    int                     ns_count;
} ctrlr_ctx_t;

static ctrlr_ctx_t g_ctrlr_ctx;

/* ───────────────────────────── 工具函数 ────────────────────────── */

static inline uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

/* 读取 /proc/self/stat utime/stime（ticks） */
static void get_proc_times(uint64_t *ut, uint64_t *st)
{
    FILE *f = fopen("/proc/self/stat", "r");
    if (!f) { *ut = *st = 0; return; }
    unsigned long u = 0, s = 0;
    fscanf(f,
        "%*d %*s %*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %lu %lu",
        &u, &s);
    fclose(f);
    *ut = u; *st = s;
}

static inline uint64_t rand_lba(ns_ctx_t *nctx)
{
    /* 随机 LBA，块大小以设备 lba_size 对齐 */
    uint32_t io_blocks = g_args.bs / nctx->lba_size;
    uint64_t max_lba   = nctx->ns_size_blocks - io_blocks;
    return ((uint64_t)rand() * rand()) % max_lba;
}

/* ─────────────────── I/O 完成回调（用户态，无内核切换）──────────── */
static void io_complete_cb(void *arg, const struct spdk_nvme_cpl *cpl)
{
    io_task_t  *task  = (io_task_t *)arg;
    ns_ctx_t   *nctx  = task->ns_ctx;
    uint64_t    now   = now_ns();
    uint64_t    lat   = now - task->submit_tsc;

    if (spdk_nvme_cpl_is_error(cpl)) {
        nctx->stats.errors++;
    } else if (g_warmup_done) {
        nctx->stats.ios_completed++;
        nctx->stats.total_latency_ns += lat;
        if (lat < nctx->stats.min_latency_ns) nctx->stats.min_latency_ns = lat;
        if (lat > nctx->stats.max_latency_ns) nctx->stats.max_latency_ns = lat;
    }

    nctx->in_flight--;

    /* 立即重新提交：若仍在运行，则提交新 I/O —— 全在用户态完成 */
    if (!g_running) return;

    uint64_t lba       = rand_lba(nctx);
    uint32_t io_blocks = g_args.bs / nctx->lba_size;
    int rc;

    task->submit_tsc = now_ns();

    if (g_args.is_write) {
        rc = spdk_nvme_ns_cmd_write(nctx->ns, nctx->qpair,
                                    task->buf, lba, io_blocks,
                                    io_complete_cb, task, 0);
    } else {
        rc = spdk_nvme_ns_cmd_read(nctx->ns, nctx->qpair,
                                   task->buf, lba, io_blocks,
                                   io_complete_cb, task, 0);
    }
    if (rc == 0) {
        nctx->in_flight++;
    } else {
        /* 提交失败时不递增 in_flight，后续轮询会补充 */
        fprintf(stderr, "re-submit failed: %d\n", rc);
    }
}

/* ─────────────────── 枚举回调：发现控制器 ─────────────────────── */
static bool probe_cb(void *cb_ctx,
                     const struct spdk_nvme_transport_id *trid,
                     struct spdk_nvme_ctrlr_opts *opts)
{
    /* 若指定了 traddr，则只接受匹配的控制器 */
    if (g_args.traddr[0] != '\0' &&
        strcmp(trid->traddr, g_args.traddr) != 0) {
        return false;
    }
    (void)cb_ctx; (void)opts;
    return true;
}

static void attach_cb(void *cb_ctx,
                      const struct spdk_nvme_transport_id *trid,
                      struct spdk_nvme_ctrlr *ctrlr,
                      const struct spdk_nvme_ctrlr_opts *opts)
{
    (void)cb_ctx; (void)trid; (void)opts;
    ctrlr_ctx_t *cctx  = &g_ctrlr_ctx;
    cctx->ctrlr        = ctrlr;

    int ns_count = spdk_nvme_ctrlr_get_num_ns(ctrlr);
    printf("控制器: %s  命名空间数: %d\n",
           spdk_nvme_ctrlr_get_regs_csts(ctrlr).bits.rdy ? "就绪" : "未就绪",
           ns_count);

    for (int i = 1; i <= ns_count && cctx->ns_count < MAX_NAMESPACES; i++) {
        struct spdk_nvme_ns *ns = spdk_nvme_ctrlr_get_ns(ctrlr, i);
        if (!ns || !spdk_nvme_ns_is_active(ns)) continue;

        ns_ctx_t *nctx = calloc(1, sizeof(ns_ctx_t));
        nctx->ns         = ns;
        nctx->lba_size   = spdk_nvme_ns_get_sector_size(ns);
        nctx->ns_size_blocks = spdk_nvme_ns_get_num_sectors(ns);
        nctx->ns_size_bytes  = nctx->ns_size_blocks * nctx->lba_size;
        nctx->stats.min_latency_ns = UINT64_MAX;

        /* 为该命名空间分配一个独立的 I/O 队列对（用户态队列）*/
        nctx->qpair = spdk_nvme_ctrlr_alloc_io_qpair(ctrlr, NULL, 0);
        if (!nctx->qpair) {
            fprintf(stderr, "无法分配 qpair，跳过 ns %d\n", i);
            free(nctx);
            continue;
        }

        /* 分配 DMA 缓冲区（从大页内存分配，无需内核参与）*/
        uint32_t io_blocks = g_args.bs / nctx->lba_size;
        for (int j = 0; j < g_args.iodepth; j++) {
            nctx->tasks[j].buf = spdk_zmalloc(
                (size_t)io_blocks * nctx->lba_size,
                0x1000,    /* 4 KiB 对齐 */
                NULL,
                SPDK_ENV_SOCKET_ID_ANY,
                SPDK_MALLOC_DMA);
            if (!nctx->tasks[j].buf) {
                fprintf(stderr, "DMA 内存分配失败\n");
                exit(EXIT_FAILURE);
            }
            nctx->tasks[j].ns_ctx = nctx;
        }

        printf("  NS %d: 大小 %.2f GiB  LBA size %u B\n",
               i,
               (double)nctx->ns_size_bytes / (1024.0*1024.0*1024.0),
               nctx->lba_size);

        cctx->ns_ctxs[cctx->ns_count++] = nctx;
    }
}

/* ──────────────────────── 信号处理 ────────────────────────────── */
static void signal_handler(int sig)
{
    (void)sig;
    g_running = 0;
}

/* ──────────────────────── 帮助信息 ────────────────────────────── */
static void usage(const char *prog)
{
    fprintf(stderr,
        "用法: %s [选项]\n"
        "  --traddr  <PCI BDF>    NVMe 设备地址，如 0000:01:00.0  (必填)\n"
        "  --rw      randread|randwrite  I/O 类型   (默认: randread)\n"
        "  --bs      <字节数>     块大小             (默认: 4096)\n"
        "  --iodepth <深度>       队列深度           (默认: 128)\n"
        "  --time    <秒>         测试时长           (默认: 10)\n"
        "  --warmup  <秒>         预热时长           (默认: 2)\n",
        prog);
}

/* ─────────────────────────────── main ─────────────────────────────── */
int main(int argc, char *argv[])
{
    /* ── 解析命令行 ── */
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--traddr")  && i+1 < argc)
            strncpy(g_args.traddr, argv[++i], sizeof(g_args.traddr)-1);
        else if (!strcmp(argv[i], "--rw")      && i+1 < argc) {
            i++;
            g_args.is_write = !strcmp(argv[i], "randwrite");
        }
        else if (!strcmp(argv[i], "--bs")      && i+1 < argc) g_args.bs       = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iodepth") && i+1 < argc) g_args.iodepth  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--time")    && i+1 < argc) g_args.time_sec = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup")  && i+1 < argc) g_args.warmup_sec = atoi(argv[++i]);
        else { usage(argv[0]); return 1; }
    }
    if (g_args.traddr[0] == '\0') { usage(argv[0]); return 1; }
    if (g_args.iodepth > MAX_IODEPTH) g_args.iodepth = MAX_IODEPTH;

    /* ── 初始化 SPDK/DPDK 环境（大页内存、PCI 设备枚举）── */
    struct spdk_env_opts env_opts;
    spdk_env_opts_init(&env_opts);
    env_opts.name       = "spdk_nvme_test";
    env_opts.shm_id     = -1;              /* 不使用共享内存 */
    if (spdk_env_init(&env_opts) < 0) {
        fprintf(stderr, "SPDK 环境初始化失败，请检查大页内存配置\n");
        return 1;
    }

    /* ── 探测并绑定 NVMe 控制器 ── */
    struct spdk_nvme_transport_id trid;
    memset(&trid, 0, sizeof(trid));
    trid.trtype  = SPDK_NVME_TRANSPORT_PCIE;
    snprintf(trid.traddr, sizeof(trid.traddr), "%s", g_args.traddr);

    printf("正在探测 NVMe 控制器 %s ...\n", g_args.traddr);
    if (spdk_nvme_probe(&trid, NULL, probe_cb, attach_cb, NULL) != 0 ||
        g_ctrlr_ctx.ns_count == 0) {
        fprintf(stderr, "未找到可用的 NVMe 命名空间，请检查 PCI 地址及驱动绑定\n");
        return 1;
    }

    printf("\nI/O 类型: %s  块大小: %u B  队列深度: %d  时长: %d s  预热: %d s\n\n",
           g_args.is_write ? "randwrite" : "randread",
           g_args.bs, g_args.iodepth, g_args.time_sec, g_args.warmup_sec);

    signal(SIGINT, signal_handler);
    srand((unsigned)time(NULL));

    /* ── 预填充：对每个命名空间提交首批 I/O ── */
    for (int n = 0; n < g_ctrlr_ctx.ns_count; n++) {
        ns_ctx_t *nctx = g_ctrlr_ctx.ns_ctxs[n];
        uint32_t  io_blocks = g_args.bs / nctx->lba_size;
        for (int j = 0; j < g_args.iodepth; j++) {
            uint64_t lba = rand_lba(nctx);
            nctx->tasks[j].submit_tsc = now_ns();
            int rc;
            if (g_args.is_write)
                rc = spdk_nvme_ns_cmd_write(nctx->ns, nctx->qpair,
                                            nctx->tasks[j].buf, lba, io_blocks,
                                            io_complete_cb, &nctx->tasks[j], 0);
            else
                rc = spdk_nvme_ns_cmd_read(nctx->ns, nctx->qpair,
                                           nctx->tasks[j].buf, lba, io_blocks,
                                           io_complete_cb, &nctx->tasks[j], 0);
            if (rc == 0) nctx->in_flight++;
            else fprintf(stderr, "初始提交失败: slot %d rc=%d\n", j, rc);
        }
    }

    /* ── 预热阶段（不计入统计）── */
    uint64_t warmup_end = now_ns() + (uint64_t)g_args.warmup_sec * 1000000000ULL;
    printf("预热中 (%d s)...\n", g_args.warmup_sec);
    while (now_ns() < warmup_end && g_running) {
        for (int n = 0; n < g_ctrlr_ctx.ns_count; n++)
            spdk_nvme_qpair_process_completions(g_ctrlr_ctx.ns_ctxs[n]->qpair, 0);
    }
    g_warmup_done = 1;

    /* ── 正式测试：重置统计并开始计时 ── */
    for (int n = 0; n < g_ctrlr_ctx.ns_count; n++) {
        ns_stats_t *s = &g_ctrlr_ctx.ns_ctxs[n]->stats;
        s->ios_completed   = 0;
        s->total_latency_ns = 0;
        s->min_latency_ns  = UINT64_MAX;
        s->max_latency_ns  = 0;
        s->errors          = 0;
    }

    uint64_t t_utime0, t_stime0;
    get_proc_times(&t_utime0, &t_stime0);
    uint64_t wall_start = now_ns();
    uint64_t deadline   = wall_start + (uint64_t)g_args.time_sec * 1000000000ULL;

    printf("正在测试...\n");

    /* ── 核心轮询循环（用户态忙轮询，无系统调用）── */
    while (g_running && now_ns() < deadline) {
        for (int n = 0; n < g_ctrlr_ctx.ns_count; n++) {
            /*
             * spdk_nvme_qpair_process_completions：
             *   - 轮询 NVMe CQ（Completion Queue）环形缓冲区
             *   - 完全在用户态（MMIO 直读），不陷入内核
             *   - 收割完成条目后直接回调 io_complete_cb
             */
            spdk_nvme_qpair_process_completions(
                g_ctrlr_ctx.ns_ctxs[n]->qpair, 0 /* 0=尽量多处理 */);
        }
    }
    g_running = 0;

    /* ── 排空残余 I/O ── */
    for (int drain = 0; drain < 100; drain++) {
        for (int n = 0; n < g_ctrlr_ctx.ns_count; n++)
            spdk_nvme_qpair_process_completions(g_ctrlr_ctx.ns_ctxs[n]->qpair, 0);
        usleep(1000);
    }

    /* ── 采集 CPU 时间 ── */
    uint64_t t_utime1, t_stime1;
    get_proc_times(&t_utime1, &t_stime1);
    uint64_t wall_end = now_ns();
    double   wall_sec = (wall_end - wall_start) / 1e9;
    long     clk_tck  = sysconf(_SC_CLK_TCK);

    double user_sec = (double)(t_utime1 - t_utime0) / clk_tck;
    double sys_sec  = (double)(t_stime1 - t_stime0) / clk_tck;

    /* ── 汇总并打印结果 ── */
    uint64_t total_ios    = 0;
    uint64_t total_lat_ns = 0;
    uint64_t min_lat_ns   = UINT64_MAX;
    uint64_t max_lat_ns   = 0;
    uint64_t total_errors = 0;

    for (int n = 0; n < g_ctrlr_ctx.ns_count; n++) {
        ns_stats_t *s = &g_ctrlr_ctx.ns_ctxs[n]->stats;
        total_ios    += s->ios_completed;
        total_lat_ns += s->total_latency_ns;
        if (s->min_latency_ns < min_lat_ns) min_lat_ns = s->min_latency_ns;
        if (s->max_latency_ns > max_lat_ns) max_lat_ns = s->max_latency_ns;
        total_errors += s->errors;
    }

    double iops    = total_ios / wall_sec;
    double bw_mbps = iops * g_args.bs / (1024.0 * 1024.0);
    double avg_lat = total_ios
                     ? (double)total_lat_ns / total_ios / 1000.0
                     : 0; /* us */
    double sys_pct  = sys_sec  / wall_sec * 100.0;
    double user_pct = user_sec / wall_sec * 100.0;

    printf("\n════════════ 测试结果 (SPDK Poll-Mode) ════════════\n");
    printf("  测试时长        : %.2f s\n",    wall_sec);
    printf("  完成 I/O 数     : %lu\n",       (unsigned long)total_ios);
    printf("  错误数          : %lu\n",       (unsigned long)total_errors);
    printf("  IOPS            : %.0f\n",      iops);
    printf("  吞吐量          : %.2f MiB/s\n", bw_mbps);
    printf("  延迟 avg/min/max: %.1f / %.1f / %.1f us\n",
           avg_lat,
           min_lat_ns == UINT64_MAX ? 0.0 : (double)min_lat_ns / 1000.0,
           (double)max_lat_ns / 1000.0);
    printf("  CPU user%%       : %.1f%%  ← 轮询核心忙等（正常现象）\n", user_pct);
    printf("  CPU sys%%        : %.1f%%  ← 几乎为 0（无内核切换）\n",   sys_pct);
    printf("  CPU total%%      : %.1f%%\n",   (user_sec + sys_sec) / wall_sec * 100.0);
    printf("═══════════════════════════════════════════════════\n");

    /* ── 清理 ── */
    for (int n = 0; n < g_ctrlr_ctx.ns_count; n++) {
        ns_ctx_t *nctx = g_ctrlr_ctx.ns_ctxs[n];
        spdk_nvme_ctrlr_free_io_qpair(nctx->qpair);
        for (int j = 0; j < g_args.iodepth; j++)
            spdk_free(nctx->tasks[j].buf);
        free(nctx);
    }
    spdk_nvme_detach(g_ctrlr_ctx.ctrlr);
    return 0;
}
