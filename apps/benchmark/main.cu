// Performance measurement of the GPU stream-collide step (lbm::step_gpu).
//
// For every selected combination of precision (float, double), grid size,
// geometry (all-fluid periodic domain, channel with walls at y = 0 and
// y = ny - 1) and collision (unforced TRT, forced TRT with Guo source) the app
// runs a warm-up, then `reps` timed repetitions of `steps` steps each, timed
// with CUDA events, and reports the median, minimum and maximum time per step.
//
// Reported per case:
//   MLUPS      = nx * ny * steps / time / 1e6  (all nodes, wall nodes included)
//   bandwidth  = bytes_per_node_update * nx * ny * steps / time
//
// bytes_per_node_update is the minimum DRAM traffic of one node update in the
// two-buffer pull scheme: Q populations read + Q populations written, each of
// sizeof(T) bytes, plus one flag byte (each node's flag has to come from DRAM
// at least once per step; the Q neighbour flag reads of a node hit lines that
// are already in cache and are not counted). So
//   bytes_per_node_update = 2 * Q * sizeof(T) + 1.
//
// Two references for the bandwidth:
//   peak  = memory clock [Hz] * 2 (double data rate) * bus width [bit] / 8,
//           from the device attributes cudaDevAttrMemoryClockRate (kHz) and
//           cudaDevAttrGlobalMemoryBusWidth;
//   copy  = measured with a simple device-to-device copy kernel (16-byte
//           elements, grid-stride loop) in this app; bytes moved = 2 * size.
//           cudaMemcpyAsync device-to-device is reported alongside.
//
// Clocks and power state are sampled with nvidia-smi (if available) at start-up
// and right after the timed repetitions of each case, when the GPU is still at
// the clocks it ran the case with.
//
// Options (all optional):
//   --sizes LIST      comma separated; N means N x N, NXxNY a rectangle.
//                     Default "auto": 128, 256, 512, ... doubling while
//                     two population buffers plus flags fit in
//                     --mem-fraction of the free device memory.
//   --mem-fraction F  fraction of free device memory a case may use (0.75).
//   --precision P     float | double | both (both).
//   --geometry G      periodic | channel | both (both).
//   --forcing F       off | on | both (both).
//   --steps N|auto    steps per timed repetition (auto: the smallest number of
//                     steps, at least 10, for which a repetition lasts at
//                     least --rep-ms, estimated from the warm-up).
//   --rep-ms X        target duration of one repetition with --steps auto (200).
//   --reps R          timed repetitions per case (5).
//   --warmup N        warm-up steps before the repetitions (= steps).
//   --nu X            kinematic viscosity (0.01).
//   --lambda X        TRT magic parameter (3/16).
//   --force X         body force component F_x for the forced cases (1e-5).
//   --copy-mib M      size of each copy-kernel buffer in MiB (256).
//   --csv FILE        also write one CSV row per case to FILE.
//   --no-smi          do not call nvidia-smi.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "lbm/collision/trt.hpp"
#include "lbm/cuda/check.hpp"
#include "lbm/cuda/device_buffer.hpp"
#include "lbm/cuda/step.cuh"
#include "lbm/force.hpp"
#include "lbm/grid.hpp"
#include "lbm/lattice/d2q9.hpp"

namespace {

using L = lbm::D2Q9;
using Grid = lbm::Grid2D<L>;

enum class Geometry { Periodic, Channel };

const char* name(Geometry g) { return g == Geometry::Periodic ? "periodic" : "channel"; }

struct Size {
    int nx, ny;
};

struct Options {
    std::vector<Size> sizes;  // empty: auto
    double mem_fraction = 0.75;
    bool do_float = true, do_double = true;
    bool do_periodic = true, do_channel = true;
    bool do_unforced = true, do_forced = true;
    int steps = 0;  // 0: auto
    double rep_ms = 200.0;
    int reps = 5;
    int warmup = -1;  // -1: equal to steps, or 10 with --steps auto
    double nu = 0.01;
    double lambda = 3.0 / 16.0;
    double force_x = 1e-5;
    std::size_t copy_mib = 256;
    std::string csv;
    bool smi = true;
};

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------
[[noreturn]] void usage_error(const std::string& msg) {
    std::fprintf(stderr, "lbm_benchmark: %s\n", msg.c_str());
    std::fprintf(stderr,
                 "usage: lbm_benchmark [--sizes LIST|auto] [--mem-fraction F]\n"
                 "       [--precision float|double|both] [--geometry periodic|channel|both]\n"
                 "       [--forcing off|on|both] [--steps N|auto] [--rep-ms X] [--reps R] [--warmup N]\n"
                 "       [--nu X] [--lambda X] [--force X] [--copy-mib M] [--csv FILE] [--no-smi]\n");
    std::exit(2);
}

int parse_int(const std::string& s, const char* what) {
    char* end = nullptr;
    const long v = std::strtol(s.c_str(), &end, 10);
    if (s.empty() || *end != '\0' || v <= 0 || v > 1000000000L)
        usage_error(std::string("invalid ") + what + " '" + s + "'");
    return static_cast<int>(v);
}

double parse_double(const std::string& s, const char* what) {
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (s.empty() || *end != '\0' || !std::isfinite(v))
        usage_error(std::string("invalid ") + what + " '" + s + "'");
    return v;
}

std::vector<Size> parse_sizes(const std::string& list) {
    std::vector<Size> sizes;
    std::size_t pos = 0;
    while (pos <= list.size()) {
        const std::size_t comma = list.find(',', pos);
        const std::string item = list.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        const std::size_t x = item.find('x');
        if (x == std::string::npos) {
            const int n = parse_int(item, "size");
            sizes.push_back({n, n});
        } else {
            sizes.push_back({parse_int(item.substr(0, x), "nx"), parse_int(item.substr(x + 1), "ny")});
        }
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return sizes;
}

Options parse_options(int argc, char** argv) {
    Options o;
    auto choose = [](const std::string& v, const char* a, const char* b, bool& fa, bool& fb, const char* what) {
        if (v == a) {
            fa = true;
            fb = false;
        } else if (v == b) {
            fa = false;
            fb = true;
        } else if (v == "both") {
            fa = fb = true;
        } else {
            usage_error(std::string("invalid ") + what + " '" + v + "'");
        }
    };
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        std::string value;
        const std::size_t eq = key.find('=');
        bool has_value = false;
        if (eq != std::string::npos) {
            value = key.substr(eq + 1);
            key = key.substr(0, eq);
            has_value = true;
        }
        if (key == "--help" || key == "-h") usage_error("help");
        if (key == "--no-smi") {
            o.smi = false;
            continue;
        }
        if (!has_value) {
            if (i + 1 >= argc) usage_error("missing value for " + key);
            value = argv[++i];
        }
        if (key == "--sizes") {
            o.sizes = (value == "auto") ? std::vector<Size>{} : parse_sizes(value);
        } else if (key == "--mem-fraction") {
            o.mem_fraction = parse_double(value, "mem-fraction");
            if (o.mem_fraction <= 0.0 || o.mem_fraction > 1.0) usage_error("mem-fraction must be in (0, 1]");
        } else if (key == "--precision") {
            choose(value, "float", "double", o.do_float, o.do_double, "precision");
        } else if (key == "--geometry") {
            choose(value, "periodic", "channel", o.do_periodic, o.do_channel, "geometry");
        } else if (key == "--forcing") {
            choose(value, "off", "on", o.do_unforced, o.do_forced, "forcing");
        } else if (key == "--steps") {
            o.steps = (value == "auto") ? 0 : parse_int(value, "steps");
        } else if (key == "--rep-ms") {
            o.rep_ms = parse_double(value, "rep-ms");
            if (o.rep_ms <= 0.0) usage_error("rep-ms must be > 0");
        } else if (key == "--reps") {
            o.reps = parse_int(value, "reps");
        } else if (key == "--warmup") {
            o.warmup = parse_int(value, "warmup");
        } else if (key == "--nu") {
            o.nu = parse_double(value, "nu");
        } else if (key == "--lambda") {
            o.lambda = parse_double(value, "lambda");
        } else if (key == "--force") {
            o.force_x = parse_double(value, "force");
        } else if (key == "--copy-mib") {
            o.copy_mib = static_cast<std::size_t>(parse_int(value, "copy-mib"));
        } else if (key == "--csv") {
            o.csv = value;
        } else {
            usage_error("unknown option " + key);
        }
    }
    if (o.warmup < 0) o.warmup = (o.steps > 0) ? o.steps : 10;
    return o;
}

// ---------------------------------------------------------------------------
// Device information and nvidia-smi sampling
// ---------------------------------------------------------------------------
struct DeviceInfo {
    std::string name;
    int sms = 0;
    int sm_clock_khz = 0;
    int mem_clock_khz = 0;
    int bus_width_bits = 0;
    int l2_bytes = 0;
    std::size_t total_bytes = 0;
    std::size_t free_bytes = 0;
    int driver = 0, runtime = 0;

    // Theoretical peak DRAM bandwidth in bytes per second:
    // memory clock [Hz] * 2 (DDR) * bus width [bit] / 8.
    double peak_bytes_per_s() const { return mem_clock_khz * 1e3 * 2.0 * bus_width_bits / 8.0; }
};

DeviceInfo query_device() {
    DeviceInfo d;
    int dev = 0;
    LBM_CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    LBM_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    d.name = prop.name;
    d.sms = prop.multiProcessorCount;
    d.total_bytes = prop.totalGlobalMem;
    d.l2_bytes = prop.l2CacheSize;
    LBM_CUDA_CHECK(cudaDeviceGetAttribute(&d.sm_clock_khz, cudaDevAttrClockRate, dev));
    LBM_CUDA_CHECK(cudaDeviceGetAttribute(&d.mem_clock_khz, cudaDevAttrMemoryClockRate, dev));
    LBM_CUDA_CHECK(cudaDeviceGetAttribute(&d.bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, dev));
    std::size_t free_b = 0, total_b = 0;
    LBM_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    d.free_bytes = free_b;
    LBM_CUDA_CHECK(cudaDriverGetVersion(&d.driver));
    LBM_CUDA_CHECK(cudaRuntimeGetVersion(&d.runtime));
    return d;
}

void print_device(const DeviceInfo& d) {
    std::printf("device: %s, %d SMs, SM clock %.0f MHz, memory clock %.0f MHz, bus %d bit, L2 %d KiB\n",
                d.name.c_str(), d.sms, d.sm_clock_khz / 1e3, d.mem_clock_khz / 1e3, d.bus_width_bits,
                d.l2_bytes / 1024);
    std::printf("memory: %.0f MiB total, %.0f MiB free; driver %d.%d, runtime %d.%d\n",
                d.total_bytes / 1048576.0, d.free_bytes / 1048576.0, d.driver / 1000, (d.driver % 1000) / 10,
                d.runtime / 1000, (d.runtime % 1000) / 10);
    std::printf("theoretical peak bandwidth: %.0f MHz * 2 * %d bit / 8 = %.1f GB/s\n", d.mem_clock_khz / 1e3,
                d.bus_width_bits, d.peak_bytes_per_s() / 1e9);
}

// One line of nvidia-smi output (clocks, power, performance state, temperature)
// or a note that it is unavailable. The output is printed verbatim.
void sample_smi(bool enabled, const char* when) {
    if (!enabled) return;
    std::FILE* p = ::popen(
        "nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,pstate,temperature.gpu "
        "--format=csv,noheader 2>/dev/null",
        "r");
    if (p == nullptr) {
        std::printf("smi [%s]: nvidia-smi unavailable\n", when);
        return;
    }
    char line[256] = {0};
    const bool ok = std::fgets(line, sizeof line, p) != nullptr;
    ::pclose(p);
    if (!ok) {
        std::printf("smi [%s]: nvidia-smi gave no output\n", when);
        return;
    }
    line[std::strcspn(line, "\r\n")] = '\0';
    std::printf("smi [%s]: sm clock, mem clock, power, pstate, temp = %s\n", when, line);
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------
class EventTimer {
public:
    EventTimer() {
        LBM_CUDA_CHECK(cudaEventCreate(&start_));
        LBM_CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~EventTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    EventTimer(const EventTimer&) = delete;
    EventTimer& operator=(const EventTimer&) = delete;

    void start() { LBM_CUDA_CHECK(cudaEventRecord(start_)); }
    // Records the stop event, waits for it and returns the elapsed milliseconds.
    double stop_ms() {
        LBM_CUDA_CHECK(cudaEventRecord(stop_));
        LBM_CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        LBM_CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return static_cast<double>(ms);
    }

private:
    cudaEvent_t start_{}, stop_{};
};

struct Stats {
    double median = 0.0, min = 0.0, max = 0.0;
    // (max - min) / median
    double spread() const { return median > 0.0 ? (max - min) / median : 0.0; }
};

Stats stats_of(std::vector<double> v) {
    if (v.empty()) throw std::logic_error("stats_of: empty sample");
    std::sort(v.begin(), v.end());
    Stats s;
    const std::size_t n = v.size();
    s.median = (n % 2 == 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    s.min = v.front();
    s.max = v.back();
    return s;
}

// ---------------------------------------------------------------------------
// Copy-kernel reference bandwidth
// ---------------------------------------------------------------------------
__global__ void copy_kernel(const uint4* __restrict__ src, uint4* __restrict__ dst, std::size_t n) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
        dst[i] = src[i];
}

struct CopyResult {
    double kernel_bytes_per_s = 0.0;  // median
    double kernel_spread = 0.0;
    double memcpy_bytes_per_s = 0.0;  // median
    double memcpy_spread = 0.0;
    std::size_t bytes = 0;  // per buffer
};

CopyResult measure_copy(const Options& opt) {
    CopyResult r;
    r.bytes = opt.copy_mib * 1048576;
    const std::size_t n = r.bytes / sizeof(uint4);
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("copy-mib too large");
    lbm::DeviceBuffer<uint4> a(static_cast<int>(n)), b(static_cast<int>(n));
    LBM_CUDA_CHECK(cudaMemset(a.data(), 0, r.bytes));
    LBM_CUDA_CHECK(cudaMemset(b.data(), 0, r.bytes));

    const unsigned block = 256;
    const unsigned grid = static_cast<unsigned>(std::min<std::size_t>((n + block - 1) / block, 1u << 16));
    EventTimer timer;
    auto launch = [&] { copy_kernel<<<grid, block>>>(a.data(), b.data(), n); LBM_CUDA_CHECK(cudaGetLastError()); };

    // warm-up
    for (int i = 0; i < 3; ++i) launch();
    LBM_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> kernel_ms, memcpy_ms;
    for (int rep = 0; rep < opt.reps; ++rep) {
        timer.start();
        launch();
        kernel_ms.push_back(timer.stop_ms());
    }
    for (int rep = 0; rep < opt.reps; ++rep) {
        timer.start();
        LBM_CUDA_CHECK(cudaMemcpyAsync(b.data(), a.data(), r.bytes, cudaMemcpyDeviceToDevice));
        memcpy_ms.push_back(timer.stop_ms());
    }
    const Stats k = stats_of(kernel_ms), m = stats_of(memcpy_ms);
    r.kernel_bytes_per_s = 2.0 * static_cast<double>(r.bytes) / (k.median * 1e-3);
    r.kernel_spread = k.spread();
    r.memcpy_bytes_per_s = 2.0 * static_cast<double>(r.bytes) / (m.median * 1e-3);
    r.memcpy_spread = m.spread();
    return r;
}

// ---------------------------------------------------------------------------
// Step benchmark
// ---------------------------------------------------------------------------
struct CaseResult {
    const char* precision;
    int nx, ny;
    Geometry geometry;
    bool forced;
    int steps, reps;
    Stats ms_per_step;
    double mlups;
    double bytes_per_s;
    double frac_peak, frac_copy;
};

template <typename T>
constexpr std::size_t bytes_per_node_update() {
    return 2 * static_cast<std::size_t>(L::Q) * sizeof(T) + 1;
}

// Two population buffers plus the flag array.
template <typename T>
std::size_t case_bytes(const Size& s) {
    const std::size_t nodes = static_cast<std::size_t>(s.nx) * static_cast<std::size_t>(s.ny);
    return nodes * (2 * static_cast<std::size_t>(L::Q) * sizeof(T) + 1);
}

std::vector<lbm::NodeType> make_flags(const Grid& g, Geometry geo) {
    std::vector<lbm::NodeType> flags(static_cast<std::size_t>(g.num_nodes()), lbm::NodeType::Fluid);
    if (geo == Geometry::Channel)
        for (int x = 0; x < g.nx(); ++x) {
            flags[static_cast<std::size_t>(g.node(x, 0))] = lbm::NodeType::Wall;
            flags[static_cast<std::size_t>(g.node(x, g.ny() - 1))] = lbm::NodeType::Wall;
        }
    return flags;
}

template <typename T, typename Collision>
CaseResult run_case(const Grid& g, Geometry geo, bool forced, const Collision& coll, const Options& opt,
                    const DeviceInfo& dev, const CopyResult& copy) {
    const int n = L::Q * g.num_nodes();
    lbm::DeviceBuffer<T> da(n), db(n);
    lbm::DeviceBuffer<lbm::NodeType> d_flags(g.num_nodes());
    LBM_CUDA_CHECK(cudaMemset(da.data(), 0, static_cast<std::size_t>(n) * sizeof(T)));
    LBM_CUDA_CHECK(cudaMemset(db.data(), 0, static_cast<std::size_t>(n) * sizeof(T)));
    const std::vector<lbm::NodeType> flags = make_flags(g, geo);
    d_flags.copy_from_host(flags.data(), g.num_nodes());

    EventTimer timer;
    LBM_CUDA_CHECK(cudaDeviceSynchronize());
    timer.start();
    for (int i = 0; i < opt.warmup; ++i) {
        lbm::step_gpu<L, T>(g, d_flags.data(), da.data(), db.data(), coll);
        std::swap(da, db);
    }
    const double warmup_ms_per_step = timer.stop_ms() / opt.warmup;

    // --steps auto: the smallest step count (at least 10) whose repetition is
    // expected to last at least opt.rep_ms, from the warm-up timing.
    int steps = opt.steps;
    if (steps == 0) {
        const double wanted = std::ceil(opt.rep_ms / warmup_ms_per_step);
        steps = static_cast<int>(std::min(std::max(wanted, 10.0), 1e6));
    }

    std::vector<double> ms_per_step;
    for (int rep = 0; rep < opt.reps; ++rep) {
        timer.start();
        for (int i = 0; i < steps; ++i) {
            lbm::step_gpu<L, T>(g, d_flags.data(), da.data(), db.data(), coll);
            std::swap(da, db);
        }
        ms_per_step.push_back(timer.stop_ms() / steps);
    }

    CaseResult r;
    r.precision = sizeof(T) == 4 ? "float" : "double";
    r.nx = g.nx();
    r.ny = g.ny();
    r.geometry = geo;
    r.forced = forced;
    r.steps = steps;
    r.reps = opt.reps;
    r.ms_per_step = stats_of(ms_per_step);
    const double nodes = static_cast<double>(g.num_nodes());
    const double s_per_step = r.ms_per_step.median * 1e-3;
    r.mlups = nodes / s_per_step / 1e6;
    r.bytes_per_s = nodes * static_cast<double>(bytes_per_node_update<T>()) / s_per_step;
    r.frac_peak = r.bytes_per_s / dev.peak_bytes_per_s();
    r.frac_copy = r.bytes_per_s / copy.kernel_bytes_per_s;
    return r;
}

void print_case(const CaseResult& r) {
    std::printf("%-6s %5dx%-5d %-8s %-8s | %6d steps x %d | %.4f ms/step (min %.4f, max %.4f, spread %.1f%%) | "
                "%8.1f MLUPS | %6.1f GB/s | %5.1f%% of peak | %5.1f%% of copy\n",
                r.precision, r.nx, r.ny, name(r.geometry), r.forced ? "forced" : "unforced", r.steps, r.reps,
                r.ms_per_step.median, r.ms_per_step.min, r.ms_per_step.max, 100.0 * r.ms_per_step.spread(),
                r.mlups, r.bytes_per_s / 1e9, 100.0 * r.frac_peak, 100.0 * r.frac_copy);
    std::fflush(stdout);
}

void write_csv_row(std::FILE* fp, const CaseResult& r) {
    if (fp == nullptr) return;
    std::fprintf(fp, "%s,%d,%d,%s,%s,%d,%d,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g\n", r.precision, r.nx, r.ny,
                 name(r.geometry), r.forced ? "forced" : "unforced", r.steps, r.reps, r.ms_per_step.median,
                 r.ms_per_step.min, r.ms_per_step.max, r.mlups, r.bytes_per_s / 1e9, r.frac_peak, r.frac_copy);
    std::fflush(fp);
}

template <typename T>
std::vector<Size> sizes_for(const Options& opt, const DeviceInfo& dev) {
    const std::size_t budget = static_cast<std::size_t>(opt.mem_fraction * static_cast<double>(dev.free_bytes));
    std::vector<Size> sizes;
    if (!opt.sizes.empty()) {
        for (const Size& s : opt.sizes) {
            if (case_bytes<T>(s) > budget)
                std::printf("skip %s %dx%d: needs %.0f MiB, budget %.0f MiB\n", sizeof(T) == 4 ? "float" : "double",
                            s.nx, s.ny, case_bytes<T>(s) / 1048576.0, budget / 1048576.0);
            else
                sizes.push_back(s);
        }
        return sizes;
    }
    for (int n = 128; n <= 1 << 20; n *= 2) {
        const Size s{n, n};
        if (case_bytes<T>(s) > budget) break;
        try {
            Grid probe(s.nx, s.ny);
        } catch (const std::invalid_argument&) {
            break;
        }
        sizes.push_back(s);
    }
    return sizes;
}

template <typename T>
void run_precision(const Options& opt, const DeviceInfo& dev, const CopyResult& copy, std::FILE* csv) {
    using Force = lbm::UniformForce<T, L::D>;
    const auto unforced = lbm::TRT<T>::template from_viscosity<L>(opt.nu, opt.lambda);
    const double fc[L::D] = {opt.force_x, 0.0};
    const auto forced =
        lbm::TRT<T, Force>::template from_viscosity<L>(opt.nu, opt.lambda, Force::create(fc));

    const std::vector<Size> sizes = sizes_for<T>(opt, dev);
    std::printf("%s: bytes per node update = 2 * %d * %zu + 1 = %zu; %zu case size(s)\n",
                sizeof(T) == 4 ? "float" : "double", L::Q, sizeof(T), bytes_per_node_update<T>(), sizes.size());
    std::vector<Geometry> geos;
    if (opt.do_periodic) geos.push_back(Geometry::Periodic);
    if (opt.do_channel) geos.push_back(Geometry::Channel);

    for (const Size& s : sizes) {
        Grid g(s.nx, s.ny);
        for (const Geometry geo : geos) {
            if (opt.do_unforced) {
                const CaseResult r = run_case<T>(g, geo, false, unforced, opt, dev, copy);
                print_case(r);
                write_csv_row(csv, r);
            }
            if (opt.do_forced) {
                const CaseResult r = run_case<T>(g, geo, true, forced, opt, dev, copy);
                print_case(r);
                write_csv_row(csv, r);
            }
        }
        sample_smi(opt.smi, (std::string(sizeof(T) == 4 ? "float " : "double ") + std::to_string(s.nx) + "x" +
                             std::to_string(s.ny))
                                .c_str());
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opt = parse_options(argc, argv);
        const DeviceInfo dev = query_device();
        print_device(dev);
        if (opt.steps > 0)
            std::printf("settings: steps %d, reps %d, warmup %d steps", opt.steps, opt.reps, opt.warmup);
        else
            std::printf("settings: steps auto (repetition >= %g ms), reps %d, warmup %d steps", opt.rep_ms, opt.reps,
                        opt.warmup);
        std::printf(", nu %g, lambda %g, F_x %g\n", opt.nu, opt.lambda, opt.force_x);
        sample_smi(opt.smi, "start");

        const CopyResult copy = measure_copy(opt);
        std::printf("copy kernel (%zu MiB per buffer): %.1f GB/s (spread %.1f%%), %.1f%% of peak; "
                    "cudaMemcpy D2D: %.1f GB/s (spread %.1f%%)\n",
                    opt.copy_mib, copy.kernel_bytes_per_s / 1e9, 100.0 * copy.kernel_spread,
                    100.0 * copy.kernel_bytes_per_s / dev.peak_bytes_per_s(), copy.memcpy_bytes_per_s / 1e9,
                    100.0 * copy.memcpy_spread);
        sample_smi(opt.smi, "after copy");

        std::FILE* csv = nullptr;
        if (!opt.csv.empty()) {
            csv = std::fopen(opt.csv.c_str(), "w");
            if (csv == nullptr) throw std::runtime_error("cannot open " + opt.csv);
            std::fprintf(csv, "precision,nx,ny,geometry,forcing,steps,reps,ms_per_step_median,ms_per_step_min,"
                              "ms_per_step_max,mlups,gb_per_s,frac_peak,frac_copy\n");
        }

        if (opt.do_float) run_precision<float>(opt, dev, copy, csv);
        if (opt.do_double) run_precision<double>(opt, dev, copy, csv);

        if (csv != nullptr) std::fclose(csv);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "lbm_benchmark: %s\n", e.what());
        return 1;
    }
}
