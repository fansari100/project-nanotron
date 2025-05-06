# Project Nanotron — Difficulty Estimation Module
# Estimates how "hard" a market decision is

from tensor import Tensor
from math import sqrt, log, exp


struct DifficultyEstimator:
    """
    Estimates the difficulty of a trading decision.
    
    High difficulty signals that the model should:
    1. Use more compute (longer MCTS search)
    2. Generate more samples (self-consistency)
    3. Be more conservative (higher confidence threshold)
    
    Difficulty indicators:
    - Volatility regime (high vol = hard)
    - Signal disagreement (conflicting indicators = hard)
    - Unusual correlation structure (regime change = hard)
    - Low liquidity (thin book = hard)
    - News/events (uncertainty = hard)
    """
    var volatility_weight: Float64
    var disagreement_weight: Float64
    var correlation_weight: Float64
    var liquidity_weight: Float64
    var event_weight: Float64
    
    # Historical baselines for normalization
    var baseline_volatility: Float64
    var baseline_spread: Float64
    var ema_alpha: Float64
    
    fn __init__(inout self):
        # Default weights (learned from historical data)
        self.volatility_weight = 0.25
        self.disagreement_weight = 0.25
        self.correlation_weight = 0.20
        self.liquidity_weight = 0.15
        self.event_weight = 0.15
        
        # Baselines
        self.baseline_volatility = 0.02  # 2% daily vol
        self.baseline_spread = 0.001     # 10 bps
        self.ema_alpha = 0.1             # EMA decay
    
    fn estimate(self, state: MarketState) -> Float64:
        """
        Estimate difficulty from current market state.
        
        Args:
            state: Current market state
            
        Returns:
            Float64: Difficulty score in [0, 1]
        """
        var difficulty = 0.0
        
        # 1. Volatility component
        let vol_score = self._volatility_score(state)
        difficulty += self.volatility_weight * vol_score
        
        # 2. Signal disagreement component
        let disagreement_score = self._disagreement_score(state)
        difficulty += self.disagreement_weight * disagreement_score
        
        # 3. Correlation structure component
        let correlation_score = self._correlation_score(state)
        difficulty += self.correlation_weight * correlation_score
        
        # 4. Liquidity component
        let liquidity_score = self._liquidity_score(state)
        difficulty += self.liquidity_weight * liquidity_score
        
        # 5. Event/news component
        let event_score = self._event_score(state)
        difficulty += self.event_weight * event_score
        
        # Clamp to [0, 1]
        return min(1.0, max(0.0, difficulty))
    
    fn _volatility_score(self, state: MarketState) -> Float64:
        """
        Score based on current vs baseline volatility.
        
        High realized vol relative to baseline = high difficulty.
        """
        var total_vol = 0.0
        for i in range(state.num_tickers):
            total_vol += Float64(state.volatility[i])
        let avg_vol = total_vol / Float64(state.num_tickers)
        
        # Ratio to baseline, capped at 3x
        let vol_ratio = min(3.0, avg_vol / self.baseline_volatility)
        
        # Normalize to [0, 1]
        return vol_ratio / 3.0
    
    fn _disagreement_score(self, state: MarketState) -> Float64:
        """
        Score based on signal disagreement.
        
        If momentum, mean-reversion, and fundamentals disagree = hard.
        """
        # In production, this would check multiple signal sources
        # For now, use order imbalance dispersion as proxy
        var mean_imbalance = 0.0
        for i in range(state.num_tickers):
            mean_imbalance += Float64(state.order_imbalance[i])
        mean_imbalance /= Float64(state.num_tickers)
        
        var variance = 0.0
        for i in range(state.num_tickers):
            let diff = Float64(state.order_imbalance[i]) - mean_imbalance
            variance += diff * diff
        variance /= Float64(state.num_tickers)
        
        let std_dev = sqrt(variance)
        
        # High std dev = high disagreement
        return min(1.0, std_dev / 0.5)
    
    fn _correlation_score(self, state: MarketState) -> Float64:
        """
        Score based on unusual correlation structure.
        
        When correlations break down or spike = regime change = hard.
        """
        # In production, this would compute rolling correlation vs historical
        # For now, use price dispersion as proxy
        var mean_price = 0.0
        for i in range(state.num_tickers):
            mean_price += Float64(state.prices[i])
        mean_price /= Float64(state.num_tickers)
        
        var variance = 0.0
        for i in range(state.num_tickers):
            let diff = Float64(state.prices[i]) - mean_price
            variance += diff * diff
        variance /= Float64(state.num_tickers)
        
        # High dispersion suggests unusual structure
        let cv = sqrt(variance) / mean_price if mean_price > 0 else 0.0
        return min(1.0, cv / 0.3)
    
    fn _liquidity_score(self, state: MarketState) -> Float64:
        """
        Score based on market liquidity.
        
        Low volume, wide spreads = hard to execute = hard decision.
        """
        var total_volume = 0.0
        for i in range(state.num_tickers):
            total_volume += Float64(state.volumes[i])
        let avg_volume = total_volume / Float64(state.num_tickers)
        
        # Low volume relative to baseline = high difficulty
        # Inverse relationship
        let volume_ratio = avg_volume / 1_000_000.0  # Baseline 1M shares
        let liquidity_score = 1.0 - min(1.0, volume_ratio)
        
        return liquidity_score
    
    fn _event_score(self, state: MarketState) -> Float64:
        """
        Score based on event/news proximity.
        
        Earnings, Fed, major news = high uncertainty = hard.
        """
        # In production, this would integrate with news/event calendar
        # For now, return 0 (no events)
        return 0.0
    
    fn update_baselines(inout self, state: MarketState):
        """
        Update rolling baselines using EMA.
        
        Called periodically to adapt to changing market conditions.
        """
        var total_vol = 0.0
        for i in range(state.num_tickers):
            total_vol += Float64(state.volatility[i])
        let avg_vol = total_vol / Float64(state.num_tickers)
        
        # EMA update
        self.baseline_volatility = (
            self.ema_alpha * avg_vol +
            (1 - self.ema_alpha) * self.baseline_volatility
        )


# Placeholder for MarketState (imported from nanotron.mojo in practice)
struct MarketState:
    var prices: Tensor[DType.float32]
    var volumes: Tensor[DType.float32]
    var order_imbalance: Tensor[DType.float32]
    var volatility: Tensor[DType.float32]
    var num_tickers: Int
    
    fn __init__(inout self, num_tickers: Int):
        self.num_tickers = num_tickers
        self.prices = Tensor[DType.float32](num_tickers)
        self.volumes = Tensor[DType.float32](num_tickers)
        self.order_imbalance = Tensor[DType.float32](num_tickers)
        self.volatility = Tensor[DType.float32](num_tickers)

