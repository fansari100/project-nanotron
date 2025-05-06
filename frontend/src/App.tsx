/**
 * Project Nanotron — React 19 Dashboard
 * WebGPU-accelerated visualization for ultra-high-frequency trading
 */

import React, { useEffect, useState, useCallback } from 'react';
import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import OrderBookViewer from './OrderBook';
import { SignalPanel, StatusPanel, PortfolioPanel } from './components';

// ============================================================================
// TYPES
// ============================================================================

interface TradingSignal {
  ticker_id: number;
  direction: number;
  confidence: number;
  size: number;
  reasoning_depth: number;
  latency_us: number;
  timestamp_ns: number;
}

interface OrderBookLevel {
  price: number;
  size: number;
  order_count: number;
}

interface OrderBook {
  symbol: string;
  timestamp_ns: number;
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
}

interface EngineStatus {
  running: boolean;
  signals_generated: number;
  orders_sent: number;
  avg_latency_us: number;
  uptime_seconds: number;
}

interface Position {
  symbol: string;
  quantity: number;
  avg_price: number;
  unrealized_pnl: number;
  realized_pnl: number;
}

interface PortfolioSummary {
  total_value: number;
  cash: number;
  positions: Position[];
  daily_pnl: number;
  sharpe_ratio: number;
}

// ============================================================================
// STORE
// ============================================================================

interface AppState {
  signals: TradingSignal[];
  orderBook: OrderBook | null;
  status: EngineStatus | null;
  portfolio: PortfolioSummary | null;
  connected: boolean;
  
  addSignal: (signal: TradingSignal) => void;
  setOrderBook: (ob: OrderBook) => void;
  setStatus: (status: EngineStatus) => void;
  setPortfolio: (portfolio: PortfolioSummary) => void;
  setConnected: (connected: boolean) => void;
}

const useStore = create<AppState>()(
  immer((set) => ({
    signals: [],
    orderBook: null,
    status: null,
    portfolio: null,
    connected: false,
    
    addSignal: (signal) => set((state) => {
      state.signals.unshift(signal);
      if (state.signals.length > 100) {
        state.signals.pop();
      }
    }),
    
    setOrderBook: (ob) => set((state) => {
      state.orderBook = ob;
    }),
    
    setStatus: (status) => set((state) => {
      state.status = status;
    }),
    
    setPortfolio: (portfolio) => set((state) => {
      state.portfolio = portfolio;
    }),
    
    setConnected: (connected) => set((state) => {
      state.connected = connected;
    }),
  }))
);

// ============================================================================
// WEBSOCKET HOOK
// ============================================================================

function useWebSocket(url: string) {
  const { addSignal, setOrderBook, setStatus, setPortfolio, setConnected } = useStore();
  
  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimeout: NodeJS.Timeout;
    
    const connect = () => {
      ws = new WebSocket(url);
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        setConnected(true);
      };
      
      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setConnected(false);
        // Reconnect after 1 second
        reconnectTimeout = setTimeout(connect, 1000);
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
      
      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          
          switch (message.type) {
            case 'Signal':
              addSignal(message);
              break;
            case 'OrderBook':
              setOrderBook(message);
              break;
            case 'Status':
              setStatus(message);
              break;
            case 'Portfolio':
              setPortfolio(message);
              break;
          }
        } catch (e) {
          console.error('Failed to parse message:', e);
        }
      };
    };
    
    connect();
    
    return () => {
      clearTimeout(reconnectTimeout);
      ws?.close();
    };
  }, [url, addSignal, setOrderBook, setStatus, setPortfolio, setConnected]);
}

// ============================================================================
// MAIN APP
// ============================================================================

export default function App() {
  const { signals, orderBook, status, portfolio, connected } = useStore();
  
  // Connect to WebSocket
  useWebSocket('ws://localhost:8080/ws');
  
  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 font-mono">
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <h1 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
              ⚡ NANOTRON
            </h1>
            <span className="text-xs text-gray-500">
              Single-Node B200 Quantitative Engine
            </span>
          </div>
          
          <div className="flex items-center gap-4">
            <ConnectionStatus connected={connected} />
            {status && (
              <div className="text-xs text-gray-400">
                Uptime: {formatDuration(status.uptime_seconds)}
              </div>
            )}
          </div>
        </div>
      </header>
      
      {/* Main Grid */}
      <main className="p-6 grid grid-cols-12 gap-6">
        {/* Left Column - Order Book & Signals */}
        <div className="col-span-4 space-y-6">
          {/* Order Book */}
          <Panel title="L3 Order Book" subtitle="WebGPU Rendered @ 120Hz">
            <OrderBookViewer orderBook={orderBook} />
          </Panel>
          
          {/* Recent Signals */}
          <Panel title="Trading Signals" subtitle="From DynScaling Engine">
            <SignalList signals={signals} />
          </Panel>
        </div>
        
        {/* Center Column - Main Visualization */}
        <div className="col-span-5 space-y-6">
          {/* Price Chart */}
          <Panel title="Market Data" subtitle="Real-time Price Action">
            <div className="h-64 flex items-center justify-center text-gray-500">
              <span>📊 Price chart renders here (deck.gl)</span>
            </div>
          </Panel>
          
          {/* MCTS Visualization */}
          <Panel title="MCTS Search Tree" subtitle="Reasoning Visualization">
            <div className="h-48 flex items-center justify-center text-gray-500">
              <span>🌳 Search tree visualization (WebGPU)</span>
            </div>
          </Panel>
        </div>
        
        {/* Right Column - Status & Portfolio */}
        <div className="col-span-3 space-y-6">
          {/* Engine Status */}
          <Panel title="Engine Status" subtitle="Performance Metrics">
            <StatusDisplay status={status} />
          </Panel>
          
          {/* Portfolio */}
          <Panel title="Portfolio" subtitle="Real-time P&L">
            <PortfolioDisplay portfolio={portfolio} />
          </Panel>
          
          {/* Latency Histogram */}
          <Panel title="Latency Distribution" subtitle="Signal Generation">
            <LatencyHistogram signals={signals} />
          </Panel>
        </div>
      </main>
    </div>
  );
}

// ============================================================================
// COMPONENTS
// ============================================================================

function Panel({ 
  title, 
  subtitle, 
  children 
}: { 
  title: string; 
  subtitle?: string; 
  children: React.ReactNode;
}) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
      <div className="px-4 py-3 border-b border-gray-800">
        <h2 className="text-sm font-semibold text-gray-200">{title}</h2>
        {subtitle && (
          <p className="text-xs text-gray-500">{subtitle}</p>
        )}
      </div>
      <div className="p-4">
        {children}
      </div>
    </div>
  );
}

function ConnectionStatus({ connected }: { connected: boolean }) {
  return (
    <div className="flex items-center gap-2">
      <div className={`w-2 h-2 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`} />
      <span className="text-xs text-gray-400">
        {connected ? 'Connected' : 'Disconnected'}
      </span>
    </div>
  );
}

function SignalList({ signals }: { signals: TradingSignal[] }) {
  if (signals.length === 0) {
    return (
      <div className="text-gray-500 text-sm text-center py-8">
        No signals yet...
      </div>
    );
  }
  
  return (
    <div className="space-y-2 max-h-64 overflow-y-auto">
      {signals.slice(0, 10).map((signal, i) => (
        <SignalRow key={i} signal={signal} />
      ))}
    </div>
  );
}

function SignalRow({ signal }: { signal: TradingSignal }) {
  const directionColor = signal.direction > 0 
    ? 'text-green-400' 
    : signal.direction < 0 
      ? 'text-red-400' 
      : 'text-gray-400';
  
  const directionText = signal.direction > 0 ? 'BUY' : signal.direction < 0 ? 'SELL' : 'HOLD';
  
  return (
    <div className="flex items-center justify-between text-xs bg-gray-800/50 rounded px-3 py-2">
      <div className="flex items-center gap-3">
        <span className={`font-bold ${directionColor}`}>{directionText}</span>
        <span className="text-gray-400">Ticker #{signal.ticker_id}</span>
      </div>
      <div className="flex items-center gap-4 text-gray-400">
        <span>Conf: {(signal.confidence * 100).toFixed(1)}%</span>
        <span>Depth: {signal.reasoning_depth}</span>
        <span>{signal.latency_us}μs</span>
      </div>
    </div>
  );
}

function StatusDisplay({ status }: { status: EngineStatus | null }) {
  if (!status) {
    return <div className="text-gray-500 text-sm">Loading...</div>;
  }
  
  return (
    <div className="space-y-3">
      <StatusRow 
        label="Status" 
        value={status.running ? '🟢 Running' : '🔴 Stopped'} 
      />
      <StatusRow 
        label="Signals Generated" 
        value={status.signals_generated.toLocaleString()} 
      />
      <StatusRow 
        label="Orders Sent" 
        value={status.orders_sent.toLocaleString()} 
      />
      <StatusRow 
        label="Avg Latency" 
        value={`${status.avg_latency_us.toFixed(2)} μs`}
        highlight 
      />
    </div>
  );
}

function StatusRow({ 
  label, 
  value, 
  highlight = false 
}: { 
  label: string; 
  value: string; 
  highlight?: boolean;
}) {
  return (
    <div className="flex justify-between text-sm">
      <span className="text-gray-400">{label}</span>
      <span className={highlight ? 'text-cyan-400 font-mono' : 'text-gray-200'}>
        {value}
      </span>
    </div>
  );
}

function PortfolioDisplay({ portfolio }: { portfolio: PortfolioSummary | null }) {
  if (!portfolio) {
    return <div className="text-gray-500 text-sm">Loading...</div>;
  }
  
  const pnlColor = portfolio.daily_pnl >= 0 ? 'text-green-400' : 'text-red-400';
  
  return (
    <div className="space-y-3">
      <StatusRow 
        label="Total Value" 
        value={`$${portfolio.total_value.toLocaleString()}`} 
      />
      <StatusRow 
        label="Cash" 
        value={`$${portfolio.cash.toLocaleString()}`} 
      />
      <div className="flex justify-between text-sm">
        <span className="text-gray-400">Daily P&L</span>
        <span className={pnlColor}>
          {portfolio.daily_pnl >= 0 ? '+' : ''}{portfolio.daily_pnl.toLocaleString()}
        </span>
      </div>
      <StatusRow 
        label="Sharpe Ratio" 
        value={portfolio.sharpe_ratio.toFixed(2)} 
        highlight
      />
    </div>
  );
}

function LatencyHistogram({ signals }: { signals: TradingSignal[] }) {
  // Group latencies into buckets
  const buckets = [0, 10, 50, 100, 500, 1000];
  const counts = buckets.map((_, i) => {
    const min = buckets[i];
    const max = buckets[i + 1] || Infinity;
    return signals.filter(s => s.latency_us >= min && s.latency_us < max).length;
  });
  
  const maxCount = Math.max(...counts, 1);
  
  return (
    <div className="space-y-2">
      {buckets.map((bucket, i) => {
        const width = (counts[i] / maxCount) * 100;
        const label = i < buckets.length - 1 
          ? `${bucket}-${buckets[i + 1]}μs`
          : `${bucket}μs+`;
        
        return (
          <div key={bucket} className="flex items-center gap-2 text-xs">
            <span className="w-16 text-gray-400">{label}</span>
            <div className="flex-1 bg-gray-800 rounded h-4">
              <div 
                className="bg-cyan-500/50 h-full rounded"
                style={{ width: `${width}%` }}
              />
            </div>
            <span className="w-8 text-right text-gray-400">{counts[i]}</span>
          </div>
        );
      })}
    </div>
  );
}

// ============================================================================
// UTILITIES
// ============================================================================

function formatDuration(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  
  if (h > 0) {
    return `${h}h ${m}m ${s}s`;
  } else if (m > 0) {
    return `${m}m ${s}s`;
  } else {
    return `${s}s`;
  }
}

