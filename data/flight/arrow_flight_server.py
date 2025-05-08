"""
Apache Arrow Flight server for high-performance data transport.

Arrow Flight uses gRPC under the hood but transmits Arrow RecordBatches
directly — achieving 10-100x throughput vs REST/JSON for columnar data.

Serves real-time market data and historical OHLCV to strategy consumers.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.flight as flight
import numpy as np
from typing import Optional


class MarketDataFlightServer(flight.FlightServerBase):
    """
    Arrow Flight server streaming market data as RecordBatches.

    Endpoints:
    - "live_ticks" → streaming L1/L2 market data
    - "ohlcv/{symbol}/{interval}" → historical OHLCV bars
    - "features/{symbol}" → pre-computed feature vectors
    """

    def __init__(self, location: str = "grpc://0.0.0.0:8815"):
        super().__init__(location)
        self._tables: dict[str, pa.Table] = {}

    def do_put(self, context, descriptor, reader, writer):
        """Ingest data from producers (e.g., tick plant, feature engine)."""
        key = descriptor.path[0].decode()
        table = reader.read_all()
        self._tables[key] = table

    def do_get(self, context, ticket):
        """Serve data to consumers (e.g., strategy, risk engine)."""
        key = ticket.ticket.decode()
        if key not in self._tables:
            raise flight.FlightUnavailableError(f"No data for key: {key}")
        table = self._tables[key]
        return flight.RecordBatchStream(table)

    def list_flights(self, context, criteria):
        """List available data streams."""
        for key, table in self._tables.items():
            descriptor = flight.FlightDescriptor.for_path(key)
            schema = table.schema
            info = flight.FlightInfo(
                schema, descriptor, [], table.num_rows, table.nbytes
            )
            yield info

    def get_flight_info(self, context, descriptor):
        key = descriptor.path[0].decode()
        if key not in self._tables:
            raise flight.FlightUnavailableError(f"Unknown: {key}")
        table = self._tables[key]
        return flight.FlightInfo(
            table.schema, descriptor, [], table.num_rows, table.nbytes
        )


class MarketDataFlightClient:
    """Client for consuming Arrow Flight data streams."""

    def __init__(self, location: str = "grpc://localhost:8815"):
        self.client = flight.connect(location)

    def get_table(self, key: str) -> pa.Table:
        """Fetch a full table by key."""
        ticket = flight.Ticket(key.encode())
        reader = self.client.do_get(ticket)
        return reader.read_all()

    def put_table(self, key: str, table: pa.Table) -> None:
        """Push a table to the server."""
        descriptor = flight.FlightDescriptor.for_path(key)
        writer, _ = self.client.do_put(descriptor, table.schema)
        writer.write_table(table)
        writer.close()

    def list_streams(self) -> list[str]:
        """List available data streams on the server."""
        return [
            info.descriptor.path[0].decode()
            for info in self.client.list_flights()
        ]

    def stream_batches(self, key: str):
        """Generator yielding RecordBatches for streaming consumption."""
        ticket = flight.Ticket(key.encode())
        reader = self.client.do_get(ticket)
        for batch in reader:
            yield batch.data


def create_sample_market_data(n_rows: int = 10_000) -> pa.Table:
    """Generate sample OHLCV data as an Arrow Table."""
    rng = np.random.default_rng(42)
    timestamps = np.arange(n_rows, dtype=np.int64) * 1_000_000  # nanoseconds
    close = 100.0 + rng.normal(0, 0.5, n_rows).cumsum()

    return pa.table({
        "timestamp_ns": timestamps,
        "open": close + rng.uniform(-0.5, 0.5, n_rows),
        "high": close + np.abs(rng.normal(0, 0.3, n_rows)),
        "low": close - np.abs(rng.normal(0, 0.3, n_rows)),
        "close": close,
        "volume": rng.integers(100, 10000, n_rows),
    })


if __name__ == "__main__":
    server = MarketDataFlightServer()
    print("Arrow Flight server running on grpc://0.0.0.0:8815")
    server.serve()
