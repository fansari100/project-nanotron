/**
 * Project Nanotron — WebGPU Order Book Visualization
 * Renders L3 order book at 120Hz using deck.gl with WebGPU backend
 */

import React, { useMemo, useRef, useEffect } from 'react';

// Types
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

interface OrderBookViewerProps {
  orderBook: OrderBook | null;
}

/**
 * WebGPU-accelerated Order Book Viewer
 *
 * In production, this would use deck.gl with @luma.gl/webgpu backend
 * to render millions of order book points at 120Hz.
 *
 * For now, we implement a CSS-based version that demonstrates the concept.
 */
export default function OrderBookViewer({ orderBook }: OrderBookViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  // Generate sample data if no order book
  const data = useMemo(() => {
    if (orderBook) {
      return orderBook;
    }
    
    // Generate sample order book for demo
    const bids: OrderBookLevel[] = [];
    const asks: OrderBookLevel[] = [];
    
    const midPrice = 100.0;
    
    for (let i = 0; i < 10; i++) {
      bids.push({
        price: midPrice - 0.01 * (i + 1),
        size: Math.floor(1000 + Math.random() * 9000),
        order_count: Math.floor(1 + Math.random() * 50),
      });
      
      asks.push({
        price: midPrice + 0.01 * (i + 1),
        size: Math.floor(1000 + Math.random() * 9000),
        order_count: Math.floor(1 + Math.random() * 50),
      });
    }
    
    return {
      symbol: 'DEMO',
      timestamp_ns: Date.now() * 1_000_000,
      bids,
      asks,
    };
  }, [orderBook]);
  
  // Calculate metrics
  const metrics = useMemo(() => {
    const midPrice = (data.bids[0].price + data.asks[0].price) / 2;
    const spread = data.asks[0].price - data.bids[0].price;
    const spreadBps = (spread / midPrice) * 10000;
    
    const totalBidSize = data.bids.reduce((sum, l) => sum + l.size, 0);
    const totalAskSize = data.asks.reduce((sum, l) => sum + l.size, 0);
    const imbalance = (totalBidSize - totalAskSize) / (totalBidSize + totalAskSize);
    
    return {
      midPrice,
      spread,
      spreadBps,
      imbalance,
      totalBidSize,
      totalAskSize,
    };
  }, [data]);
  
  // Max size for scaling bars
  const maxSize = useMemo(() => {
    const allSizes = [...data.bids, ...data.asks].map(l => l.size);
    return Math.max(...allSizes);
  }, [data]);
  
  return (
    <div className="space-y-4">
      {/* Metrics Bar */}
      <div className="flex justify-between text-xs">
        <div className="flex gap-4">
          <Metric label="Mid" value={metrics.midPrice.toFixed(2)} />
          <Metric label="Spread" value={`${metrics.spreadBps.toFixed(1)} bps`} />
        </div>
        <div className="flex gap-4">
          <Metric 
            label="Imbalance" 
            value={`${(metrics.imbalance * 100).toFixed(1)}%`}
            color={metrics.imbalance > 0 ? 'text-green-400' : 'text-red-400'}
          />
        </div>
      </div>
      
      {/* Order Book Grid */}
      <div className="grid grid-cols-2 gap-4">
        {/* Bids (Left) */}
        <div className="space-y-1">
          <div className="text-xs text-gray-500 mb-2">BIDS</div>
          {data.bids.map((level, i) => (
            <OrderBookRow
              key={i}
              level={level}
              maxSize={maxSize}
              side="bid"
            />
          ))}
        </div>
        
        {/* Asks (Right) */}
        <div className="space-y-1">
          <div className="text-xs text-gray-500 mb-2">ASKS</div>
          {data.asks.map((level, i) => (
            <OrderBookRow
              key={i}
              level={level}
              maxSize={maxSize}
              side="ask"
            />
          ))}
        </div>
      </div>
      
      {/* Depth Chart Placeholder */}
      <div className="h-24 bg-gray-800/50 rounded flex items-center justify-center">
        <span className="text-xs text-gray-500">
          Depth chart (WebGPU @ 120Hz)
        </span>
      </div>
      
      {/* WebGPU Info */}
      <WebGPUStatus />
    </div>
  );
}

function OrderBookRow({
  level,
  maxSize,
  side,
}: {
  level: OrderBookLevel;
  maxSize: number;
  side: 'bid' | 'ask';
}) {
  const width = (level.size / maxSize) * 100;
  const bgColor = side === 'bid' ? 'bg-green-500/20' : 'bg-red-500/20';
  const textColor = side === 'bid' ? 'text-green-400' : 'text-red-400';
  
  return (
    <div className="relative h-6 bg-gray-800/30 rounded overflow-hidden">
      {/* Size bar */}
      <div
        className={`absolute inset-y-0 ${side === 'bid' ? 'right-0' : 'left-0'} ${bgColor}`}
        style={{ width: `${width}%` }}
      />
      
      {/* Content */}
      <div className="relative flex items-center justify-between px-2 h-full text-xs">
        <span className={textColor}>{level.price.toFixed(2)}</span>
        <span className="text-gray-400">{level.size.toLocaleString()}</span>
      </div>
    </div>
  );
}

function Metric({
  label,
  value,
  color = 'text-cyan-400',
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="flex gap-1">
      <span className="text-gray-500">{label}:</span>
      <span className={color}>{value}</span>
    </div>
  );
}

function WebGPUStatus() {
  const [supported, setSupported] = React.useState<boolean | null>(null);
  
  useEffect(() => {
    // Check WebGPU support
    const checkWebGPU = async () => {
      if ('gpu' in navigator) {
        try {
          const adapter = await (navigator as any).gpu.requestAdapter();
          setSupported(adapter !== null);
        } catch {
          setSupported(false);
        }
      } else {
        setSupported(false);
      }
    };
    
    checkWebGPU();
  }, []);
  
  return (
    <div className="flex items-center gap-2 text-xs text-gray-500">
      <span>WebGPU:</span>
      {supported === null && <span>Checking...</span>}
      {supported === true && <span className="text-green-400">✓ Supported</span>}
      {supported === false && <span className="text-yellow-400">⚠ Fallback to WebGL</span>}
    </div>
  );
}

// ============================================================================
// WEBGPU RENDERER (Placeholder for production implementation)
// ============================================================================

/**
 * Production WebGPU implementation would:
 *
 * 1. Create GPUDevice and GPURenderPipeline
 * 2. Upload order book data to GPU buffer
 * 3. Render points/bars in single draw call
 * 4. Use compute shaders for aggregation
 * 5. Target 120fps with millions of points
 *
 * Example deck.gl setup:
 *
 * import DeckGL from '@deck.gl/react';
 * import { ScatterplotLayer } from '@deck.gl/layers';
 * import { WebGPUDevice } from '@luma.gl/webgpu';
 *
 * const device = await WebGPUDevice.create();
 *
 * <DeckGL
 *   device={device}
 *   layers={[
 *     new ScatterplotLayer({
 *       id: 'orderbook',
 *       data: orderBookPoints,
 *       getPosition: d => [d.price, d.level],
 *       getRadius: d => Math.sqrt(d.size),
 *       getFillColor: d => d.side === 'bid' ? [0, 255, 0] : [255, 0, 0],
 *     })
 *   ]}
 * />
 */

