# Project Nanotron — DynScaling Module
# Dynamic Compute Allocation based on Task Difficulty

from tensor import Tensor


@register_passable("trivial")
struct DifficultyLevel:
    """Enum for difficulty levels."""
    var value: Int
    
    alias EASY = DifficultyLevel(0)
    alias MEDIUM = DifficultyLevel(1)
    alias HARD = DifficultyLevel(2)
    
    fn __init__(inout self, value: Int):
        self.value = value
    
    fn __eq__(self, other: DifficultyLevel) -> Bool:
        return self.value == other.value
    
    fn __str__(self) -> String:
        if self.value == 0:
            return "EASY"
        elif self.value == 1:
            return "MEDIUM"
        else:
            return "HARD"


struct ComputeBudget:
    """Compute budget for a given difficulty level."""
    var simulations: Int
    var samples: Int
    var max_depth: Int
    var timeout_us: Int
    
    fn __init__(inout self, level: DifficultyLevel):
        if level == DifficultyLevel.EASY:
            self.simulations = 1
            self.samples = 1
            self.max_depth = 1
            self.timeout_us = 10
        elif level == DifficultyLevel.MEDIUM:
            self.simulations = 100
            self.samples = 4
            self.max_depth = 8
            self.timeout_us = 100
        else:  # HARD
            self.simulations = 10000
            self.samples = 64
            self.max_depth = 32
            self.timeout_us = 1000


struct DynScaler:
    """
    Dynamic Compute Scaler
    
    Implements the DynScaling approach from research:
    - Estimates task difficulty
    - Allocates compute budget accordingly
    - Adapts based on uncertainty feedback
    
    Reference: "DynScaling: Efficient Verifier-free Inference Scaling"
    """
    var easy_threshold: Float64
    var hard_threshold: Float64
    var history_window: Int
    var uncertainty_history: Tensor[DType.float64]
    var accuracy_history: Tensor[DType.float64]
    var current_idx: Int
    
    fn __init__(
        inout self,
        easy_threshold: Float64 = 0.3,
        hard_threshold: Float64 = 0.7,
        history_window: Int = 1000
    ):
        self.easy_threshold = easy_threshold
        self.hard_threshold = hard_threshold
        self.history_window = history_window
        self.uncertainty_history = Tensor[DType.float64](history_window)
        self.accuracy_history = Tensor[DType.float64](history_window)
        self.current_idx = 0
    
    fn get_difficulty_level(self, difficulty: Float64) -> DifficultyLevel:
        """
        Map continuous difficulty score to discrete level.
        
        Args:
            difficulty: Continuous difficulty score in [0, 1]
            
        Returns:
            DifficultyLevel: EASY, MEDIUM, or HARD
        """
        if difficulty < self.easy_threshold:
            return DifficultyLevel.EASY
        elif difficulty < self.hard_threshold:
            return DifficultyLevel.MEDIUM
        else:
            return DifficultyLevel.HARD
    
    fn get_compute_budget(self, level: DifficultyLevel) -> ComputeBudget:
        """
        Get compute budget for given difficulty level.
        
        Args:
            level: DifficultyLevel (EASY, MEDIUM, HARD)
            
        Returns:
            ComputeBudget: Simulations, samples, depth, timeout
        """
        return ComputeBudget(level)
    
    fn adaptive_budget(
        self,
        difficulty: Float64,
        available_time_us: Int
    ) -> ComputeBudget:
        """
        Compute adaptive budget based on difficulty AND available time.
        
        This implements the bandit-based allocation from DynScaling paper.
        
        Args:
            difficulty: Continuous difficulty score
            available_time_us: Available time in microseconds
            
        Returns:
            ComputeBudget: Optimized for given constraints
        """
        let level = self.get_difficulty_level(difficulty)
        var budget = ComputeBudget(level)
        
        # Scale down if time constrained
        if budget.timeout_us > available_time_us:
            let scale = Float64(available_time_us) / Float64(budget.timeout_us)
            budget.simulations = max(1, int(Float64(budget.simulations) * scale))
            budget.samples = max(1, int(Float64(budget.samples) * scale))
            budget.max_depth = max(1, int(Float64(budget.max_depth) * scale))
            budget.timeout_us = available_time_us
        
        return budget
    
    fn update_history(
        inout self,
        uncertainty: Float64,
        was_correct: Bool
    ):
        """
        Update history for adaptive threshold learning.
        
        This enables online learning of optimal thresholds.
        
        Args:
            uncertainty: Model's uncertainty for this decision
            was_correct: Whether the trading decision was correct
        """
        self.uncertainty_history[self.current_idx] = uncertainty
        self.accuracy_history[self.current_idx] = 1.0 if was_correct else 0.0
        self.current_idx = (self.current_idx + 1) % self.history_window
    
    fn calibrate_thresholds(inout self):
        """
        Recalibrate thresholds based on historical performance.
        
        Implements the insight from SCALE paper:
        "Allocate more compute where it helps most"
        """
        # Compute accuracy per uncertainty bin
        var bin_accuracy = Tensor[DType.float64](10)
        var bin_counts = Tensor[DType.int32](10)
        
        for i in range(self.history_window):
            let uncertainty = self.uncertainty_history[i]
            let accuracy = self.accuracy_history[i]
            let bin_idx = min(9, int(uncertainty * 10))
            bin_accuracy[bin_idx] += accuracy
            bin_counts[bin_idx] += 1
        
        # Find optimal thresholds (where accuracy drops)
        for i in range(10):
            if bin_counts[i] > 0:
                bin_accuracy[i] /= Float64(bin_counts[i])
        
        # Update thresholds based on accuracy curve
        # Easy = where accuracy > 90%
        # Hard = where accuracy < 70%
        for i in range(10):
            if bin_accuracy[i] < 0.9 and self.easy_threshold > Float64(i) / 10:
                self.easy_threshold = Float64(i) / 10
                break
        
        for i in range(9, -1, -1):
            if bin_accuracy[i] > 0.7 and self.hard_threshold < Float64(i) / 10:
                self.hard_threshold = Float64(i) / 10
                break


struct SelfConsistencyVoter:
    """
    Self-Consistency Voting for Hard Decisions
    
    Reference: "Self-Consistency Improves Chain of Thought Reasoning"
    
    For hard decisions, generate K samples and vote.
    """
    var num_samples: Int
    var temperature: Float64
    
    fn __init__(inout self, num_samples: Int = 64, temperature: Float64 = 0.7):
        self.num_samples = num_samples
        self.temperature = temperature
    
    fn vote(
        self,
        signals: Tensor[DType.int32]  # K signals, each in {-1, 0, 1}
    ) -> Tuple[Int, Float64]:
        """
        Majority vote over K samples.
        
        Args:
            signals: Tensor of K trading signals
            
        Returns:
            Tuple[Int, Float64]: (majority_signal, confidence)
        """
        var counts = Tensor[DType.int32](3)  # sell, hold, buy
        
        for i in range(signals.num_elements()):
            let signal = signals[i]
            let idx = signal + 1  # -1 -> 0, 0 -> 1, 1 -> 2
            counts[idx] += 1
        
        # Find majority
        var max_count = 0
        var max_idx = 1  # Default to hold
        for i in range(3):
            if counts[i] > max_count:
                max_count = counts[i]
                max_idx = i
        
        let majority_signal = max_idx - 1  # 0 -> -1, 1 -> 0, 2 -> 1
        let confidence = Float64(max_count) / Float64(signals.num_elements())
        
        return (majority_signal, confidence)

