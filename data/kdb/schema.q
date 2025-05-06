/ Project Nanotron — KDB+ Schema
/ Market data tables optimized for GPUDirect access

/ ============================================================================
/ TABLE SCHEMAS
/ ============================================================================

/ Trade table - tick-by-tick execution data
trade:([]
    time:`timestamp$();        / Nanosecond timestamp
    sym:`symbol$();            / Symbol
    price:`float$();           / Trade price
    size:`int$();              / Trade size
    side:`char$();             / 'B' or 'S'
    exch:`symbol$();           / Exchange
    seq:`long$()               / Sequence number
);

/ Quote table - best bid/ask updates
quote:([]
    time:`timestamp$();
    sym:`symbol$();
    bid:`float$();
    bidsz:`int$();
    ask:`float$();
    asksz:`int$();
    exch:`symbol$();
    seq:`long$()
);

/ Order book depth - Level 3 data
orderbook:([]
    time:`timestamp$();
    sym:`symbol$();
    level:`int$();             / 0-9 (10 levels)
    bidpx:`float$();
    bidsz:`int$();
    bidcnt:`int$();            / Number of orders at level
    askpx:`float$();
    asksz:`int$();
    askcnt:`int$();
    seq:`long$()
);

/ Signal output from engine
signal:([]
    time:`timestamp$();
    sym:`symbol$();
    direction:`int$();         / -1, 0, 1
    confidence:`float$();
    size:`float$();
    reasoning_depth:`int$();
    latency_us:`long$()
);

/ Position tracking
position:([]
    time:`timestamp$();
    sym:`symbol$();
    qty:`long$();
    avg_price:`float$();
    unrealized_pnl:`float$();
    realized_pnl:`float$()
);

/ ============================================================================
/ ATTRIBUTES FOR PERFORMANCE
/ ============================================================================

/ Apply sorted attribute to time column for fast time-based queries
`trade set `time xasc trade;
`quote set `time xasc quote;
`orderbook set `time xasc orderbook;

/ Apply grouped attribute to symbol for fast symbol lookups
update `g#sym from `trade;
update `g#sym from `quote;
update `g#sym from `orderbook;

/ ============================================================================
/ PARTITIONING
/ ============================================================================

/ Date-partitioned historical data
/ .Q.dpft[`:hdb; .z.D; `sym; `trade]

/ ============================================================================
/ HELPER FUNCTIONS
/ ============================================================================

/ Get latest quote for symbol
getQuote:{[s]
    select last bid, last bidsz, last ask, last asksz 
    from quote where sym=s
    };

/ Get OHLCV bars
getOHLCV:{[s;start;end;bucket]
    select 
        o:first price, 
        h:max price, 
        l:min price, 
        c:last price, 
        v:sum size
    by sym, bucket xbar time 
    from trade 
    where sym=s, time within (start;end)
    };

/ Get order book snapshot
getOrderBook:{[s]
    select 
        level, 
        bidpx, bidsz, bidcnt, 
        askpx, asksz, askcnt
    from orderbook 
    where sym=s, time=max time
    };

/ Calculate order imbalance
orderImbalance:{[s;levels]
    t:getOrderBook[s];
    t:select from t where level<levels;
    totBid:exec sum bidsz from t;
    totAsk:exec sum asksz from t;
    (totBid-totAsk) % (totBid+totAsk)
    };

/ Calculate VWAP
vwap:{[s;start;end]
    exec (sum price*size) % sum size 
    from trade 
    where sym=s, time within (start;end)
    };

/ ============================================================================
/ STREAMING AGGREGATES
/ ============================================================================

/ Rolling statistics maintained in real-time
rolling:([]
    sym:`symbol$();
    window:`int$();            / Window size in seconds
    mean_price:`float$();
    std_price:`float$();
    mean_volume:`float$();
    vwap:`float$();
    imbalance:`float$()
);

/ Update rolling stats on new trade
updateRolling:{[t]
    s:t`sym;
    now:t`time;
    
    / 1-second window
    t1:select from trade where sym=s, time>now-0D00:00:01;
    if[count t1;
        mean1:avg t1`price;
        std1:dev t1`price;
        vol1:avg t1`size;
        vwap1:vwap[s;now-0D00:00:01;now];
        imb1:orderImbalance[s;5];
        
        `rolling upsert (s;1;mean1;std1;vol1;vwap1;imb1)
    ];
    
    / 5-second window
    t5:select from trade where sym=s, time>now-0D00:00:05;
    if[count t5;
        mean5:avg t5`price;
        std5:dev t5`price;
        vol5:avg t5`size;
        vwap5:vwap[s;now-0D00:00:05;now];
        imb5:orderImbalance[s;10];
        
        `rolling upsert (s;5;mean5;std5;vol5;vwap5;imb5)
    ];
    };

/ ============================================================================
/ INITIALIZATION
/ ============================================================================

\l gpu_loader.q

.z.ts:{
    / Heartbeat - runs every 100ms
    / Update any time-based aggregates here
    };

system "t 100";  / Timer every 100ms

-1 "Nanotron KDB+ Schema Loaded";

