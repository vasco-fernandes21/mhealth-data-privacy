from opacus.accountants import RDPAccountant


def estimate_epsilon(sigma: float, sample_rate: float, epochs_or_steps: int, delta: float = 1e-5, is_steps: bool = False) -> float:
    """
    Estimates the privacy budget (epsilon) without running training.
    
    Args:
        sigma: Noise multiplier
        sample_rate: Batch size / dataset size
        epochs_or_steps: Either number of epochs (if is_steps=False) or number of steps (if is_steps=True)
        delta: Target delta (default 1e-5)
        is_steps: If True, epochs_or_steps is already the number of steps. If False, convert epochs to steps.
    
    Returns:
        Estimated epsilon value
    """
    if sigma <= 0:
        return 0.0
    
    if sample_rate <= 0 or sample_rate > 1:
        return float('inf')
    
    try:
        accountant = RDPAccountant()
        
        if is_steps:
            steps = int(epochs_or_steps)
        else:
            steps = int(epochs_or_steps / sample_rate) if sample_rate > 0 else epochs_or_steps
        
        steps = max(1, steps)
        
        # Opacus RDP tracking
        for _ in range(steps):
            accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)

        epsilon = accountant.get_epsilon(delta=delta)
        
        # Validate epsilon
        if not isinstance(epsilon, (int, float)) or epsilon == float('inf') or epsilon != epsilon:
            return 0.0
            
        return float(epsilon)
    except Exception:
        return 0.0

