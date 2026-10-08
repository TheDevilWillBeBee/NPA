import torch

@torch.no_grad()
def relative_error(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Compute relative absolute error between two tensors

    Args:
        a: First tensor
        b: Second tensor
        eps: Small epsilon to avoid division by zero

    Returns:
        Relative error as a scalar
    """
    if a is None or b is None:
        return 0.0
    
    if isinstance(a, tuple):
        return sum([relative_error(x, y, eps) for (x, y) in zip(a, b)])
    abs_diff = torch.abs(a - b)
    abs_ref = torch.abs(a) + eps
    rel_error = (abs_diff / abs_ref).mean()
    return rel_error.item()


def benchmark_function_forward(func, *args, num_runs=100, warmup_runs=25):
    """
    Benchmark a function by running it multiple times and measuring average execution time
    using CUDA events.

    Args:
        func: Function to benchmark
        *args: Arguments to pass to the function
        num_runs: Number of timing runs
        warmup_runs: Number of warmup runs (not timed)

    Returns:
        Average execution time in milliseconds
    """
    # Warmup runs
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = func(*args)
    torch.cuda.synchronize()

    # Timed runs
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()

    for _ in range(num_runs):
        with torch.no_grad():
            _ = func(*args)

    end_event.record()
    torch.cuda.synchronize()

    avg_time_ms = start_event.elapsed_time(end_event) / num_runs
    return avg_time_ms


def loss_fn(x):
    return x.sum()

def benchmark_function_backward(func, loss_fn, *args, num_runs=100, warmup_runs=25):
    """
    Benchmark a function's backward pass by running it multiple times and measuring average execution time
    using CUDA events.
    
    Args:
        func: Function to benchmark
        *args: Arguments to pass to the function (must include tensors with requires_grad=True)
        num_runs: Number of timing runs
        warmup_runs: Number of warmup runs (not timed)
    
    Returns:
        Average execution time in milliseconds
    """
    # Warmup runs
    for _ in range(warmup_runs):
        args_grad = []
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.requires_grad:
                args_grad.append(arg.clone().detach().requires_grad_(True))
            else:
                args_grad.append(arg)
        
        output = func(*args_grad)
        
        # Handle both single tensor and tuple of tensors
        if isinstance(output, tuple):
            grad_outputs = tuple(torch.randn_like(o) for o in output)
            torch.autograd.backward(output, grad_tensors=grad_outputs)
        else:
            grad_output = torch.randn_like(output)
            output.backward(gradient=grad_output)
    
    torch.cuda.synchronize()
    
    if isinstance(output, tuple):
        grad_outputs = tuple(torch.randn_like(o) for o in output)
    else:
        grad_output = torch.randn_like(output)
    
    # Timed runs - only timing the backward pass
    events = []
    for _ in range(num_runs):
        args_grad = []
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.requires_grad:
                args_grad.append(arg.clone().detach().requires_grad_(True))
            else:
                args_grad.append(arg)
        
        # Forward pass (not timed)
        output = func(*args_grad)
        
        # Handle both single tensor and tuple of tensors
        
        
        # Time only the backward pass
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        if isinstance(output, tuple):
            # For tuple outputs, use torch.autograd.backward with grad_tensors
            torch.autograd.backward(output, grad_tensors=grad_outputs)
        else:
            output.backward(gradient=grad_output)
        end_event.record()
        
        events.append((start_event, end_event))
    
    # Synchronize once at the end and calculate total time
    torch.cuda.synchronize()
    total_backward_time = sum(start.elapsed_time(end) for start, end in events)
    avg_backward_time_ms = total_backward_time / num_runs
    
    return avg_backward_time_ms