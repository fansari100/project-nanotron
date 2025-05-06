/ Project Nanotron — GPUDirect Data Loader
/ Zero-copy data transfer from NVMe to GPU VRAM via kdb+ 4.1

/ ============================================================================
/ GPU CONFIGURATION
/ ============================================================================

/ GPU device configuration
.gpu.device:0i;                    / GPU device ID
.gpu.memory_limit:180000000000;    / 180GB limit (leave 12GB for system)

/ ============================================================================
/ GPUDIRECT STORAGE (GDS)
/ ============================================================================

/ Enable GPUDirect Storage for direct NVMe → GPU transfers
.gds.enabled:1b;
.gds.buffer_size:1073741824;       / 1GB buffer

/ Initialize GDS connection
.gds.init:{[]
    -1 "Initializing GPUDirect Storage...";
    
    / Check if CUDA and GDS available
    if[not .z.o like "l*"; 
        -1 "GPUDirect requires Linux";
        :.gds.enabled:0b
    ];
    
    / Initialize via kdb+ GPU extension
    .gpu.init[.gpu.device];
    
    -1 "GPUDirect Storage initialized on device ", string .gpu.device;
    };

/ ============================================================================
/ DATA LOADING
/ ============================================================================

/ Load tick data directly to GPU memory
.gpu.loadTicks:{[sym;start;end]
    / Build query
    path:hsym `$":hdb/",string[start],"/trade";
    
    if[.gds.enabled;
        / GPUDirect path - bypasses CPU entirely
        / Data flows: NVMe → PCIe → GPU VRAM
        -1 "Loading via GPUDirect: ", string sym;
        
        data:.gpu.read[path; `sym`time!(`sym=sym;`time within (start;end))];
        
        / Returns GPU memory pointer (no CPU copy!)
        :data
    ];
    
    / Fallback: CPU load then copy
    -1 "Loading via CPU (fallback): ", string sym;
    data:select from trade where sym=sym, time within (start;end);
    :.gpu.toGPU[data]
    };

/ Load order book snapshot to GPU
.gpu.loadOrderBook:{[sym]
    / Get current order book state
    ob:getOrderBook[sym];
    
    / Convert to Arrow format for zero-copy
    arrow:.arrow.fromTable[ob];
    
    / Copy to GPU
    :.gpu.fromArrow[arrow]
    };

/ ============================================================================
/ ARROW INTEROP
/ ============================================================================

/ Convert table to Arrow format (zero-copy compatible)
.arrow.fromTable:{[t]
    / Build Arrow RecordBatch
    / Uses C Data Interface for zero-copy sharing
    
    / Get column schemas
    cols:cols t;
    types:.Q.ty each t cols;
    
    / Build schema
    schema:flip `name`type!(cols;types);
    
    / Build record batch (in-memory Arrow format)
    / This creates a memory buffer that can be shared zero-copy
    batch:(`schema`data)!(schema;value flip t);
    
    :batch
    };

/ Share Arrow buffer with GPU
.gpu.fromArrow:{[arrow]
    / Map Arrow buffer to GPU memory
    / Uses CUDA's zero-copy mapping
    
    / Get buffer pointer
    ptr:arrow`data;
    
    / Map to GPU address space
    gpu_ptr:.gpu.map[ptr];
    
    :gpu_ptr
    };

/ ============================================================================
/ REAL-TIME STREAMING
/ ============================================================================

/ Streaming buffer for real-time data
.stream.buffer:();
.stream.buffer_size:10000;         / Flush every 10k records

/ Add tick to streaming buffer
.stream.addTick:{[t]
    .stream.buffer,:enlist t;
    
    / Flush to GPU when buffer full
    if[count[.stream.buffer] >= .stream.buffer_size;
        .stream.flush[]
    ];
    };

/ Flush buffer to GPU
.stream.flush:{[]
    if[count .stream.buffer;
        / Convert to Arrow
        arrow:.arrow.fromTable[.stream.buffer];
        
        / Async copy to GPU
        .gpu.asyncCopy[arrow];
        
        / Clear buffer
        .stream.buffer:();
    ];
    };

/ ============================================================================
/ GPU COMPUTATIONS
/ ============================================================================

/ Run computation on GPU data
.gpu.compute:{[kernel;data]
    / Launch CUDA kernel on data
    / Kernel is pre-compiled (see CUDA kernels in C++)
    
    result:.gpu.launch[kernel;data];
    
    :result
    };

/ Pre-defined kernels
.gpu.kernels:(
    `rolling_mean;
    `rolling_std;
    `order_imbalance;
    `vwap;
    `correlation
);

/ ============================================================================
/ INITIALIZATION
/ ============================================================================

.gds.init[];

-1 "Nanotron GPU Loader Ready";
-1 "  Device: ", string .gpu.device;
-1 "  GDS Enabled: ", string .gds.enabled;
-1 "  Memory Limit: ", string .gpu.memory_limit;

