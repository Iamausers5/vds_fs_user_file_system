/**
 * @file  vdb_fs_types.h
 * @brief VDB-FS core type definitions, constants, and error-handling macros.
 *
 * This header is the single source of truth for all fundamental layout
 * parameters of the GPU-resident filesystem.  Every module includes this
 * header transitively via <vdb_fs/vdb_fs.h>.
 *
 * Memory Layout of the 256 MiB GPU Pinned Region
 * ┌─────────────────────────────────────────────────────────────┐  0x0000_0000
 * │  Superblock (4 KiB)                                        │
 * ├─────────────────────────────────────────────────────────────┤  0x0000_1000
 * │  Inode Table  (64 MiB)                                     │
 * │    - max 262 144 inodes × 256 B each                       │
 * ├─────────────────────────────────────────────────────────────┤  0x0400_1000
 * │  Hash Table / Directory Index (128 MiB)                    │
 * │    - open-addressing, GPU-parallel probing                 │
 * ├─────────────────────────────────────────────────────────────┤  0x0C40_1000
 * │  Block Bitmap  (16 MiB)                                    │
 * │    - 1 bit per 4 KiB block → tracks 512 GiB address space  │
 * ├─────────────────────────────────────────────────────────────┤  0x0E40_1000
 * │  Metadata Cache (48 KiB)  — hot-path inode/dir cache       │
 * ├─────────────────────────────────────────────────────────────┤  0x0E40_D000
 * │  Journal Ring Buffer (remaining ~27.95 MiB)                │
 * │    - write-ahead log for crash consistency                 │
 * └─────────────────────────────────────────────────────────────┘  0x1000_0000
 */

#ifndef VDB_FS_TYPES_H
#define VDB_FS_TYPES_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>

/* ── Portability: host / device annotations ────────────────────────────────── */
#ifdef __CUDACC__
  #include <cuda_runtime.h>
  #define VDB_HOST        __host__
  #define VDB_DEVICE      __device__
  #define VDB_HOST_DEVICE __host__ __device__
  #define VDB_GLOBAL      __global__
#else
  #define VDB_HOST
  #define VDB_DEVICE
  #define VDB_HOST_DEVICE
  #define VDB_GLOBAL
#endif

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  Section 1 : Compile-time Constants                                       */
/* ═══════════════════════════════════════════════════════════════════════════ */

namespace vdb {
namespace fs {

/* ── GPU Pinned Region ─────────────────────────────────────────────────────── */
/** Total GPU pinned memory allocated via vdb_alloc_pinned (bytes). */
static constexpr uint64_t GPU_PINNED_REGION_SIZE      = 256ULL * 1024 * 1024;   // 256 MiB

/* ── Sub-region Sizes ──────────────────────────────────────────────────────── */
static constexpr uint64_t SUPERBLOCK_SIZE              = 4ULL * 1024;            //   4 KiB
static constexpr uint64_t INODE_TABLE_SIZE             = 64ULL * 1024 * 1024;    //  64 MiB
static constexpr uint64_t HASH_TABLE_SIZE              = 128ULL * 1024 * 1024;   // 128 MiB
static constexpr uint64_t BLOCK_BITMAP_SIZE            = 16ULL * 1024 * 1024;    //  16 MiB
static constexpr uint64_t METADATA_CACHE_SIZE          = 48ULL * 1024;           //  48 KiB

/** Journal occupies whatever remains after the fixed regions. */
static constexpr uint64_t JOURNAL_SIZE                 =
    GPU_PINNED_REGION_SIZE
    - SUPERBLOCK_SIZE
    - INODE_TABLE_SIZE
    - HASH_TABLE_SIZE
    - BLOCK_BITMAP_SIZE
    - METADATA_CACHE_SIZE;

/* Sanity: make sure the layout fits. */
static_assert(
    SUPERBLOCK_SIZE + INODE_TABLE_SIZE + HASH_TABLE_SIZE +
    BLOCK_BITMAP_SIZE + METADATA_CACHE_SIZE + JOURNAL_SIZE
        == GPU_PINNED_REGION_SIZE,
    "GPU pinned region layout overflow");

/* ── Sub-region Offsets (byte offset from region base) ─────────────────────── */
static constexpr uint64_t OFF_SUPERBLOCK               = 0;
static constexpr uint64_t OFF_INODE_TABLE               = OFF_SUPERBLOCK    + SUPERBLOCK_SIZE;
static constexpr uint64_t OFF_HASH_TABLE                = OFF_INODE_TABLE   + INODE_TABLE_SIZE;
static constexpr uint64_t OFF_BLOCK_BITMAP              = OFF_HASH_TABLE    + HASH_TABLE_SIZE;
static constexpr uint64_t OFF_METADATA_CACHE            = OFF_BLOCK_BITMAP  + BLOCK_BITMAP_SIZE;
static constexpr uint64_t OFF_JOURNAL                   = OFF_METADATA_CACHE + METADATA_CACHE_SIZE;

/* ── Inode / Block Parameters ──────────────────────────────────────────────── */
static constexpr uint32_t INODE_SIZE                    = 256;                   // bytes per inode
static constexpr uint32_t MAX_INODES                    =
    static_cast<uint32_t>(INODE_TABLE_SIZE / INODE_SIZE);                        // 262 144

static constexpr uint32_t BLOCK_SIZE                    = 4096;                  // 4 KiB data block
static constexpr uint32_t BLOCK_SIZE_SHIFT              = 12;                    // log2(4096)

/** Maximum addressable data blocks tracked by the bitmap. */
static constexpr uint64_t MAX_DATA_BLOCKS               =
    BLOCK_BITMAP_SIZE * 8;                                                       // 134 217 728

/** Maximum addressable storage (blocks × block_size). */
static constexpr uint64_t MAX_STORAGE_BYTES             =
    MAX_DATA_BLOCKS * BLOCK_SIZE;                                                // 512 GiB

/* ── Path / Name Limits ────────────────────────────────────────────────────── */
static constexpr uint32_t MAX_FILENAME_LEN              = 255;
static constexpr uint32_t MAX_PATH_LEN                  = 4096;
static constexpr uint32_t MAX_OPEN_FILES                = 65536;

/* ── NVMe I/O Parameters ──────────────────────────────────────────────────── */
static constexpr uint32_t NVME_QUEUE_DEPTH              = 256;
static constexpr uint32_t NVME_MAX_IO_SIZE              = 128 * 1024;            // 128 KiB per NVMe cmd
static constexpr uint32_t NVME_SECTOR_SIZE              = 512;

/* ── Hash Table Tuning ─────────────────────────────────────────────────────── */
static constexpr uint32_t HASH_BUCKET_STRIDE            = 64;                   // bytes, cache-line aligned
static constexpr uint32_t HASH_MAX_PROBE                = 32;                   // linear probe limit
static constexpr float    HASH_LOAD_FACTOR_MAX          = 0.75f;

/* ── Journal ───────────────────────────────────────────────────────────────── */
static constexpr uint32_t JOURNAL_ENTRY_ALIGN           = 64;                   // byte alignment

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  Section 2 : Fundamental Types                                            */
/* ═══════════════════════════════════════════════════════════════════════════ */

/** Inode number (0 = invalid / sentinel). */
using ino_t  = uint32_t;
/** Logical block number. */
using blk_t  = uint64_t;
/** File descriptor handle returned to userspace. */
using fd_t   = int32_t;

static constexpr ino_t INVALID_INO = 0;
static constexpr fd_t  INVALID_FD  = -1;

/* ── Error Codes ───────────────────────────────────────────────────────────── */

/** VDB-FS error codes (negative = error, 0 = success). */
enum class VdbStatus : int32_t {
    OK              =  0,
    ERR_IO          = -1,   // NVMe / DMA transfer failure
    ERR_NOMEM       = -2,   // GPU or host memory exhaustion
    ERR_NOENT       = -3,   // file / inode not found
    ERR_EXIST       = -4,   // file already exists
    ERR_INVAL       = -5,   // invalid argument
    ERR_NOSPC       = -6,   // no free blocks / inodes
    ERR_NAMETOOLONG = -7,   // filename exceeds MAX_FILENAME_LEN
    ERR_NOTDIR      = -8,   // expected directory, got file
    ERR_ISDIR       = -9,   // expected file, got directory
    ERR_BUSY        = -10,  // resource in use
    ERR_PERM        = -11,  // permission denied
    ERR_CORRUPT     = -12,  // metadata inconsistency detected
    ERR_CUDA        = -100, // CUDA runtime error
    ERR_SPDK        = -101, // SPDK subsystem error
    ERR_GDRCOPY     = -102, // GDRCopy mapping error
};

/** Convert VdbStatus to human-readable string. */
VDB_HOST_DEVICE inline const char* vdb_status_str(VdbStatus s) {
    switch (s) {
        case VdbStatus::OK:              return "OK";
        case VdbStatus::ERR_IO:          return "I/O error";
        case VdbStatus::ERR_NOMEM:       return "out of memory";
        case VdbStatus::ERR_NOENT:       return "not found";
        case VdbStatus::ERR_EXIST:       return "already exists";
        case VdbStatus::ERR_INVAL:       return "invalid argument";
        case VdbStatus::ERR_NOSPC:       return "no space left";
        case VdbStatus::ERR_NAMETOOLONG: return "name too long";
        case VdbStatus::ERR_NOTDIR:      return "not a directory";
        case VdbStatus::ERR_ISDIR:       return "is a directory";
        case VdbStatus::ERR_BUSY:        return "resource busy";
        case VdbStatus::ERR_PERM:        return "permission denied";
        case VdbStatus::ERR_CORRUPT:     return "metadata corrupt";
        case VdbStatus::ERR_CUDA:        return "CUDA error";
        case VdbStatus::ERR_SPDK:        return "SPDK error";
        case VdbStatus::ERR_GDRCOPY:     return "GDRCopy error";
        default:                         return "unknown error";
    }
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  Section 3 : File Types & Inode Flags                                     */
/* ═══════════════════════════════════════════════════════════════════════════ */

enum class FileType : uint8_t {
    REGULAR   = 0x01,
    DIRECTORY = 0x02,
    SYMLINK   = 0x03,
    VECTOR_DB = 0x10,   // opaque blob: vector index / HNSW graph
};

/** Inode flags (bitmask). */
enum InodeFlags : uint32_t {
    IFLAG_NONE      = 0,
    IFLAG_IMMUTABLE = 1 << 0,   // cannot modify after creation
    IFLAG_SYNC      = 1 << 1,   // force journal flush on every write
    IFLAG_P2P       = 1 << 2,   // eligible for GPU-SSD P2P DMA
    IFLAG_PINNED    = 1 << 3,   // keep data resident in GPU memory
};

} // namespace fs
} // namespace vdb

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  Section 4 : Error-Handling Macros (host side)                            */
/* ═══════════════════════════════════════════════════════════════════════════ */

/**
 * CUDA_CHECK(expr)
 * Wraps any CUDA runtime call. On failure, logs file:line and aborts.
 * In debug builds, additionally synchronizes the device to catch async errors.
 */
#ifdef __CUDACC__
#define CUDA_CHECK(expr)                                                         \
    do {                                                                         \
        cudaError_t _err = (expr);                                               \
        if (__builtin_expect(_err != cudaSuccess, 0)) {                          \
            std::fprintf(stderr,                                                 \
                "[VDB-FS CUDA FATAL] %s:%d  %s  →  %s (%d)\n",                  \
                __FILE__, __LINE__, #expr,                                       \
                cudaGetErrorString(_err), static_cast<int>(_err));               \
            std::abort();                                                        \
        }                                                                        \
    } while (0)

/**
 * CUDA_CHECK_LAST()
 * Checks the last asynchronous CUDA error.
 * Typically called after kernel launches: kernel<<<...>>>(); CUDA_CHECK_LAST();
 */
#define CUDA_CHECK_LAST()                                                        \
    do {                                                                         \
        cudaError_t _err = cudaGetLastError();                                   \
        if (__builtin_expect(_err != cudaSuccess, 0)) {                          \
            std::fprintf(stderr,                                                 \
                "[VDB-FS CUDA FATAL] %s:%d  cudaGetLastError → %s (%d)\n",       \
                __FILE__, __LINE__,                                              \
                cudaGetErrorString(_err), static_cast<int>(_err));               \
            std::abort();                                                        \
        }                                                                        \
    } while (0)
#endif /* __CUDACC__ */

/**
 * SPDK_CHECK(expr, msg)
 * Wraps SPDK calls that return 0 on success or -errno on failure.
 */
#define SPDK_CHECK(expr, msg)                                                    \
    do {                                                                         \
        int _rc = (expr);                                                        \
        if (__builtin_expect(_rc != 0, 0)) {                                     \
            std::fprintf(stderr,                                                 \
                "[VDB-FS SPDK FATAL] %s:%d  %s  →  rc=%d (%s)\n",               \
                __FILE__, __LINE__, (msg), _rc, strerror(-_rc));                 \
            std::abort();                                                        \
        }                                                                        \
    } while (0)

/**
 * GDR_CHECK(expr, msg)
 * Wraps GDRCopy calls that return 0 on success.
 */
#define GDR_CHECK(expr, msg)                                                     \
    do {                                                                         \
        int _rc = (expr);                                                        \
        if (__builtin_expect(_rc != 0, 0)) {                                     \
            std::fprintf(stderr,                                                 \
                "[VDB-FS GDR FATAL] %s:%d  %s  →  rc=%d\n",                     \
                __FILE__, __LINE__, (msg), _rc);                                 \
            std::abort();                                                        \
        }                                                                        \
    } while (0)

/**
 * VDB_CHECK(status)
 * Asserts a VdbStatus is OK; otherwise logs and aborts.
 */
#define VDB_CHECK(status)                                                        \
    do {                                                                         \
        ::vdb::fs::VdbStatus _s = (status);                                      \
        if (__builtin_expect(_s != ::vdb::fs::VdbStatus::OK, 0)) {              \
            std::fprintf(stderr,                                                 \
                "[VDB-FS FATAL] %s:%d  status=%d (%s)\n",                        \
                __FILE__, __LINE__,                                              \
                static_cast<int>(_s), ::vdb::fs::vdb_status_str(_s));           \
            std::abort();                                                        \
        }                                                                        \
    } while (0)

/**
 * VDB_LOG(level, fmt, ...)
 * Lightweight logging. Define VDB_LOG_LEVEL (0=OFF,1=ERR,2=WARN,3=INFO,4=DBG).
 */
#ifndef VDB_LOG_LEVEL
  #ifdef VDB_DEBUG
    #define VDB_LOG_LEVEL 4
  #else
    #define VDB_LOG_LEVEL 2
  #endif
#endif

#define VDB_LOG(level, fmt, ...)                                                  \
    do {                                                                          \
        if ((level) <= VDB_LOG_LEVEL) {                                           \
            static constexpr const char* _tags[] =                                \
                {"", "ERR", "WARN", "INFO", "DBG"};                              \
            std::fprintf(stderr, "[VDB-FS %s] %s:%d  " fmt "\n",                \
                _tags[(level)], __FILE__, __LINE__, ##__VA_ARGS__);              \
        }                                                                        \
    } while (0)

#define VDB_ERR(fmt, ...)   VDB_LOG(1, fmt, ##__VA_ARGS__)
#define VDB_WARN(fmt, ...)  VDB_LOG(2, fmt, ##__VA_ARGS__)
#define VDB_INFO(fmt, ...)  VDB_LOG(3, fmt, ##__VA_ARGS__)
#define VDB_DBG(fmt, ...)   VDB_LOG(4, fmt, ##__VA_ARGS__)

/* ── Likely / Unlikely branch hints ────────────────────────────────────────── */
#define VDB_LIKELY(x)   __builtin_expect(!!(x), 1)
#define VDB_UNLIKELY(x) __builtin_expect(!!(x), 0)

#endif /* VDB_FS_TYPES_H */
