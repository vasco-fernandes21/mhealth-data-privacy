from opacus.accountants import RDPAccountant


def estimate_epsilon(sigma: float, sample_rate: float, epochs: int, delta: float = 1e-5) -> float:
    """
    Estimates the privacy budget (epsilon) without running training.
    """
    if sigma <= 0:
        return float('inf')
    
    if sample_rate <= 0 or sample_rate > 1:
        return float('inf')
    
    try:
        accountant = RDPAccountant()
        steps = int(epochs / sample_rate) if sample_rate > 0 else epochs
        steps = max(1, steps)
        
        # Opacus RDP tracking
        for _ in range(steps):
            accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)

        epsilon = accountant.get_epsilon(delta=delta)
        
        # Validate epsilon
        if not isinstance(epsilon, (int, float)) or epsilon == float('inf') or epsilon != epsilon:
            return 0.0
            
        return float(epsilon)
    except Exception as e:
        print(f"Warning: Epsilon estimation failed: {e}")
        return 0.0

