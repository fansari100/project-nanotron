/**
 * Project Nanotron — Shared Memory Interface
 * Lock-free SPSC queue for inter-process communication
 *
 * Used for:
 * - Mojo/JAX → C++ signal passing
 * - C++ → Rust/Frontend data streaming
 */

#pragma once

#include <cstdint>
#include <atomic>
#include <cstring>

namespace nanotron {

// ============================================================================
// CACHE LINE ALIGNMENT
// ============================================================================

constexpr size_t CACHE_LINE_SIZE = 64;

template<typename T>
struct alignas(CACHE_LINE_SIZE) CacheAligned {
    T value;
    
    CacheAligned() : value{} {}
    explicit CacheAligned(T v) : value(v) {}
    
    operator T&() { return value; }
    operator const T&() const { return value; }
};

// ============================================================================
// TRADING SIGNAL
// ============================================================================

#pragma pack(push, 1)
struct TradingSignal {
    uint32_t ticker_id;
    int8_t direction;     // -1 = sell, 0 = hold, 1 = buy
    uint8_t padding[3];
    float confidence;
    float size;
    int32_t reasoning_depth;
    int64_t latency_us;
    uint64_t timestamp_ns;
    
    static constexpr size_t SIZE = 32;
};
#pragma pack(pop)

static_assert(sizeof(TradingSignal) == TradingSignal::SIZE, 
              "TradingSignal must be 32 bytes");

// ============================================================================
// RING BUFFER HEADER
// ============================================================================

struct alignas(CACHE_LINE_SIZE) RingBufferHeader {
    // Producer writes here (Mojo/JAX)
    CacheAligned<std::atomic<uint64_t>> write_pos;
    
    // Consumer reads here (C++)
    CacheAligned<std::atomic<uint64_t>> read_pos;
    
    // Buffer capacity (must be power of 2)
    uint64_t capacity;
    
    // Magic number for validation
    uint64_t magic;
    
    static constexpr uint64_t MAGIC = 0x4E414E4F54524F4EULL;  // "NANOTRON"
};

// ============================================================================
// LOCK-FREE SPSC QUEUE
// ============================================================================

template<typename T, size_t Capacity>
class SPSCQueue {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    
public:
    SPSCQueue() : head_(0), tail_(0) {
        static_assert(sizeof(T) <= CACHE_LINE_SIZE, "Element too large");
    }
    
    /**
     * Try to push element (producer only).
     *
     * @returns true if pushed, false if full
     */
    [[nodiscard]] bool try_push(const T& item) noexcept {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        const size_t next_tail = (current_tail + 1) & (Capacity - 1);
        
        // Check if full
        if (next_tail == head_.load(std::memory_order_acquire)) {
            return false;
        }
        
        // Write element
        buffer_[current_tail] = item;
        
        // Publish
        tail_.store(next_tail, std::memory_order_release);
        
        return true;
    }
    
    /**
     * Try to pop element (consumer only).
     *
     * @returns true if popped, false if empty
     */
    [[nodiscard]] bool try_pop(T& item) noexcept {
        const size_t current_head = head_.load(std::memory_order_relaxed);
        
        // Check if empty
        if (current_head == tail_.load(std::memory_order_acquire)) {
            return false;
        }
        
        // Read element
        item = buffer_[current_head];
        
        // Advance head
        head_.store((current_head + 1) & (Capacity - 1), std::memory_order_release);
        
        return true;
    }
    
    /**
     * Check if empty (approximate, may race).
     */
    [[nodiscard]] bool empty() const noexcept {
        return head_.load(std::memory_order_relaxed) == 
               tail_.load(std::memory_order_relaxed);
    }
    
    /**
     * Get approximate size (may race).
     */
    [[nodiscard]] size_t size() const noexcept {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t tail = tail_.load(std::memory_order_relaxed);
        return (tail - head) & (Capacity - 1);
    }

private:
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> head_;
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> tail_;
    alignas(CACHE_LINE_SIZE) std::array<T, Capacity> buffer_;
};

// ============================================================================
// MARKET DATA STRUCTURES
// ============================================================================

#pragma pack(push, 1)
struct MarketDataUpdate {
    uint64_t timestamp_ns;
    uint32_t symbol_id;
    uint8_t update_type;   // 0 = trade, 1 = quote
    uint8_t side;          // 0 = bid, 1 = ask
    uint8_t padding[2];
    double price;
    int32_t size;
    uint32_t sequence_num;
    
    static constexpr size_t SIZE = 40;
};
#pragma pack(pop)

static_assert(sizeof(MarketDataUpdate) == MarketDataUpdate::SIZE,
              "MarketDataUpdate must be 40 bytes");

#pragma pack(push, 1)
struct OrderBookLevel {
    double price;
    int32_t size;
    int32_t order_count;
};
#pragma pack(pop)

static_assert(sizeof(OrderBookLevel) == 16, "OrderBookLevel must be 16 bytes");

struct OrderBook {
    static constexpr size_t MAX_LEVELS = 10;
    
    uint64_t timestamp_ns;
    uint32_t symbol_id;
    uint32_t sequence_num;
    
    std::array<OrderBookLevel, MAX_LEVELS> bids;
    std::array<OrderBookLevel, MAX_LEVELS> asks;
    
    [[nodiscard]] double mid_price() const noexcept {
        return (bids[0].price + asks[0].price) / 2.0;
    }
    
    [[nodiscard]] double spread() const noexcept {
        return asks[0].price - bids[0].price;
    }
    
    [[nodiscard]] double order_imbalance() const noexcept {
        int32_t total_bid = 0;
        int32_t total_ask = 0;
        
        for (size_t i = 0; i < MAX_LEVELS; ++i) {
            total_bid += bids[i].size;
            total_ask += asks[i].size;
        }
        
        double total = static_cast<double>(total_bid + total_ask);
        if (total < 1e-8) return 0.0;
        
        return static_cast<double>(total_bid - total_ask) / total;
    }
};

}  // namespace nanotron

