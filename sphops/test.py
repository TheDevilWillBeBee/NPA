import torch
from dataclasses import dataclass
from typing import Callable, Tuple, List, Any, Optional

from sph import (
    blur,
    density,
    count,
    gradient,
    density_gradient,
    moment_matrix,
    HashGrid,
    bin_particles,
)
from torch_sph import density as density_torch, blur as blur_torch
from torch_sph import gradient as gradient_torch, count as count_torch
from torch_sph import bin_particles as bin_particles_torch
from torch_sph import density_gradient as density_gradient_torch
from torch_sph import moment_matrix as moment_matrix_torch
from utils import (
    relative_error,
    benchmark_function_forward,
    benchmark_function_backward,
)


def loss_fn(x):
    """Default loss function for backward pass benchmarking"""
    if isinstance(x, tuple):
        return sum((out * out).sum() for out in x)
    return (x * x).sum()


@dataclass
class KernelConfig:
    """Configuration for a single kernel test"""

    name: str
    cuda_func: Callable
    torch_func: Callable
    args_func: Callable  # Function that returns args tuple for the kernel
    benchmark: bool = True
    has_backward: bool = False  # Whether to benchmark backward pass


class CUDAKernelValidator:
    """Validates and benchmarks CUDA kernels against PyTorch reference implementations"""

    def __init__(
        self,
        grid_size,
        boundary,
        mode,
        B: int,
        N: int,
        C: int,
        dim: int,
        eps: float,
        sigma: float,
        max_particles_per_block=64,
        rand_type: str = "randn",
        device: str = "cuda:0",
        num_benchmark_runs: int = 512,
        warmup_runs: int = 128,
        seed: Optional[int] = 42,
    ):
        self.B = B
        self.N = N
        self.C = C
        self.dim = dim
        self.eps = eps
        self.sigma = sigma
        self.max_particles_per_block = max_particles_per_block
        self.rand_type = rand_type
        self.device = torch.device(device)
        self.num_benchmark_runs = num_benchmark_runs
        self.warmup_runs = warmup_runs
        self.seed = seed
        self.grid_size = grid_size
        self.boundary = boundary
        self.mode = mode  # "ParticleBased" or "GridBased"
        self.periodic = str(boundary).lower() == "periodic"
        if isinstance(grid_size, (list, tuple)):
            self.box_size = [g * self.eps for g in grid_size]
        else:
            self.box_size = self.grid_size * self.eps

        # Set random seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Data containers
        self.x = None
        self.s = None
        self.mass = None
        self.grid = None
        self.snapshot = None
        self.x_bin = None
        self.s_bin = None
        self.mass_bin = None
        self.rho = None

        self._setup_data()

    def _setup_data(self):
        """Initialize test data and grid"""
        with torch.no_grad():
            if self.rand_type == "randn":
                self.x = (
                    torch.randn(
                        self.B, self.N, 2 if self.dim == 2 else 4, device=self.device
                    )
                    * self.sigma
                )
            elif self.rand_type == "rand":
                self.x = (
                    (
                        torch.rand(
                            self.B,
                            self.N,
                            2 if self.dim == 2 else 4,
                            device=self.device,
                        )
                        - 0.5
                    )
                    * 2.0
                    * self.sigma
                )
            else:
                raise ValueError(f"Unknown rand_type: {self.rand_type}")

            self.s = torch.randn(self.B, self.N, self.C, device=self.device)
            self.mass = torch.rand(self.B, self.N, device=self.device) + 1e-3

            self.grid = HashGrid(
                dim=self.dim,
                boundary=self.boundary,
                # boundary="clamped",
                
                mode=self.mode,
                grid_size=[self.grid_size] * self.dim,
                eps=self.eps,
                num_particles=self.N,
                batch_size=self.B,
                max_particles_per_block=self.max_particles_per_block,
            )

            self.snapshot = self.grid.instantiate(self.x)
            self.x_bin, self.s_bin, self.mass_bin = bin_particles(
                self.x, self.s, self.snapshot, self.mass
            )
            self.rho = density(self.x_bin, self.snapshot, self.mass_bin)

    def _define_kernels(self) -> List[KernelConfig]:
        """Define all kernels to test - easily extensible"""
        return [
            KernelConfig(
                name="bin_particles",
                cuda_func=bin_particles,
                torch_func=bin_particles_torch,
                args_func=lambda: (self.x, self.s, self.snapshot),
                benchmark=True,
                has_backward=True,
            ),
            KernelConfig(
                name="count",
                cuda_func=count,
                torch_func=count_torch,
                args_func=lambda: (self.x_bin, self.snapshot),
                benchmark=True,
                has_backward=False,
            ),
            KernelConfig(
                name="density",
                cuda_func=density,
                torch_func=density_torch,
                args_func=lambda: (self.x_bin, self.snapshot, self.mass_bin),
                benchmark=True,
                has_backward=True,
            ),
            KernelConfig(
                name="blur",
                cuda_func=blur,
                torch_func=blur_torch,
                args_func=lambda: (self.x_bin, self.rho, self.s_bin, self.snapshot, self.mass_bin),
                benchmark=True,
                has_backward=True,
            ),
            KernelConfig(
                name="gradient",
                cuda_func=gradient,
                torch_func=gradient_torch,
                args_func=lambda: (self.x_bin, self.rho, self.s_bin, self.snapshot, self.mass_bin),
                benchmark=True,
                has_backward=True,
            ),
            KernelConfig(
                name="density_gradient",
                cuda_func=density_gradient,
                torch_func=density_gradient_torch,
                args_func=lambda: (self.x_bin, self.snapshot, self.mass_bin),
                benchmark=True,
                has_backward=True,
            ),
            KernelConfig(
                name="moment_matrix",
                cuda_func=moment_matrix,
                torch_func=moment_matrix_torch,
                args_func=lambda: (self.x_bin, self.rho, self.snapshot, self.mass_bin),
                benchmark=True,
                has_backward=True,
            ),
        ]

    def _define_special_benchmarks(self) -> List[Tuple[str, Callable, Tuple]]:
        """Define special benchmark cases (non-kernel functions)"""
        return [
            #             ("bin_particles", bin_particles, (self.x, self.s, self.snapshot)),
            ("instantiate", self.grid.instantiate, (self.x, )),
        ]

    def validate_kernel(
        self, kernel: KernelConfig
    ) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
        """Validate a single kernel and return forward and backward relative errors"""
        with torch.no_grad():
            cuda_args = kernel.args_func()

            # Get CUDA result
            cuda_result = kernel.cuda_func(*cuda_args)

            # Get PyTorch reference result
            # Convert args for PyTorch version (extract position and use density directly)
            if kernel.name in ["blur", "gradient"]:
                x_pos = cuda_args[0][..., : self.dim]
                rho = cuda_args[1]
                mass = cuda_args[4]
                torch_args = (
                    x_pos,
                    rho,
                    cuda_args[2],
                    self.eps,
                    mass,
                    self.periodic,
                    self.box_size if self.periodic else None,
                )
            elif kernel.name == "moment_matrix":
                x_pos = cuda_args[0][..., : self.dim]
                rho = cuda_args[1]
                mass = cuda_args[3]
                torch_args = (
                    x_pos,
                    rho,
                    self.eps,
                    mass,
                    self.periodic,
                    self.box_size if self.periodic else None,
                )
            elif kernel.name in ["count", "density", "density_gradient"]:
                x_pos = cuda_args[0][..., : self.dim]
                if kernel.name == "count":
                    torch_args = (
                        x_pos,
                        self.eps,
                        self.periodic,
                        self.box_size if self.periodic else None,
                    )
                else:
                    mass = cuda_args[2]
                    torch_args = (
                        x_pos,
                        self.eps,
                        mass,
                        self.periodic,
                        self.box_size if self.periodic else None,
                    )
            elif kernel.name == "bin_particles":
                torch_args = (cuda_args[0], cuda_args[1], cuda_args[2].permutation)
            else:
                # Default case - assume same arguments
                torch_args = cuda_args

            torch_result = kernel.torch_func(*torch_args)

            # Compute forward relative error
            forward_error = relative_error(cuda_result, torch_result)

        # Validate backward if applicable
        backward_error = None
        if kernel.has_backward:
            # Prepare SEPARATE cloned args with gradient tracking for CUDA version
            cuda_args_list = list(kernel.args_func())
            cuda_x = cuda_args_list[0].clone().detach().requires_grad_(True)
            cuda_args_grad = [cuda_x]

            # Handle remaining args based on kernel type
            if kernel.name in ["blur", "gradient"]:
                # Args are: x_bin, rho, s_bin, snapshot, mass
                cuda_rho = cuda_args_list[1].clone().detach().requires_grad_(True)
                cuda_s = cuda_args_list[2].clone().detach().requires_grad_(True)
                cuda_mass = cuda_args_list[4].detach()
                cuda_args_grad.extend([cuda_rho, cuda_s, cuda_args_list[3], cuda_mass])
            elif kernel.name == "moment_matrix":
                cuda_rho = cuda_args_list[1].clone().detach().requires_grad_(True)
                cuda_mass = cuda_args_list[3].detach()
                cuda_args_grad.extend([cuda_rho, cuda_args_list[2], cuda_mass])
            elif kernel.name in ["density", "density_gradient"]:
                cuda_mass = cuda_args_list[2].detach()
                cuda_args_grad.extend([cuda_args_list[1], cuda_mass])
            elif kernel.name in ["bin_particles"]:
                cuda_s = cuda_args_list[1].clone().detach().requires_grad_(True)
                cuda_args_grad.extend([cuda_s, cuda_args_list[2]])
            else:
                # For other kernels, add remaining args
                for arg in cuda_args_list[1:]:
                    if isinstance(arg, torch.Tensor) and arg.dtype.is_floating_point:
                        cuda_args_grad.append(arg.clone().detach().requires_grad_(True))
                    else:
                        cuda_args_grad.append(arg)

            # Prepare SEPARATE cloned args for PyTorch version
            if kernel.name in ["blur", "gradient"]:
                # For torch version: x_pos, rho, s, eps
                torch_x_pos = (
                    cuda_x[..., : self.dim].clone().detach().requires_grad_(True)
                )
                torch_rho = cuda_rho.clone().detach().requires_grad_(True)
                torch_s = cuda_s.clone().detach().requires_grad_(True)
                torch_args_grad = [
                    torch_x_pos,
                    torch_rho,
                    torch_s,
                    self.eps,
                    cuda_mass,
                    self.periodic,
                    self.box_size if self.periodic else None,
                ]
            elif kernel.name in ["count", "density", "density_gradient"]:
                torch_x_pos = (
                    cuda_x[..., : self.dim].clone().detach().requires_grad_(True)
                )
                if kernel.name == "count":
                    torch_args_grad = [
                        torch_x_pos,
                        self.eps,
                        self.periodic,
                        self.box_size if self.periodic else None,
                    ]
                else:
                    torch_args_grad = [
                        torch_x_pos,
                        self.eps,
                        cuda_mass,
                        self.periodic,
                        self.box_size if self.periodic else None,
                    ]
            elif kernel.name == "moment_matrix":
                torch_x_pos = (
                    cuda_x[..., : self.dim].clone().detach().requires_grad_(True)
                )
                torch_rho = cuda_args_list[1].clone().detach().requires_grad_(True)
                torch_args_grad = [
                    torch_x_pos,
                    torch_rho,
                    self.eps,
                    cuda_mass,
                    self.periodic,
                    self.box_size if self.periodic else None,
                ]

            elif kernel.name == "bin_particles":
                torch_args_grad = [
                    (
                        arg.clone().detach().requires_grad_(True)
                        if isinstance(arg, torch.Tensor) and arg.dtype.is_floating_point
                        else arg.permutation
                    )
                    for arg in cuda_args_grad
                ]
                torch_x_pos = torch_args_grad[0]
                torch_s = torch_args_grad[1]
            else:
                print("Please implement")

            # Forward pass
            cuda_output = kernel.cuda_func(*cuda_args_grad)
            torch_output = kernel.torch_func(*torch_args_grad)

            # Backward pass using loss function
            cuda_loss = loss_fn(cuda_output)
            torch_loss = loss_fn(torch_output)

            cuda_loss.backward()
            torch_loss.backward()

            # Compare gradients for all relevant tensors
            max_error = 0.0

            if kernel.name in [
                "blur",
                "gradient",
                "density",
                "bin_particles",
                "density_gradient",
                "moment_matrix",
            ]:
                cuda_x_grad = cuda_x.grad
                torch_x_grad = torch_x_pos.grad
                if cuda_x_grad is not None and torch_x_grad is not None:
                    # Only compare the position dimensions
                    error_x = relative_error(
                        cuda_x_grad[..., : self.dim], torch_x_grad[..., : self.dim]
                    )
                    max_error = max(max_error, error_x)
                    # print("Error x:", error_x)

            if kernel.name in ["blur", "gradient", "bin_particles"]:
                # Compare s gradients
                cuda_s_grad = cuda_s.grad
                torch_s_grad = torch_s.grad
                if cuda_s_grad is not None and torch_s_grad is not None:
                    error_s = relative_error(cuda_s_grad, torch_s_grad)
                    #                     if kernel.name == "bin_particles":
                    #                         print(cuda_s_grad[0], torch_s_grad[0])
                    max_error = max(max_error, error_s)

            if kernel.name in ["blur", "gradient", "moment_matrix"]:
                # Compare rho gradients
                cuda_rho_grad = cuda_rho.grad
                torch_rho_grad = torch_rho.grad
                if cuda_rho_grad is not None and torch_rho_grad is not None:
                    error_rho = relative_error(cuda_rho_grad, torch_rho_grad)
                    max_error = max(max_error, error_rho)

            backward_error = max_error

        return forward_error, backward_error, cuda_result, torch_result

    def benchmark_kernel(
        self, name: str, func: Callable, args: Tuple, benchmark_backward: bool = False
    ) -> Tuple[float, Optional[float]]:
        """Benchmark a single kernel (forward and optionally backward)"""
        forward_time = benchmark_function_forward(
            func, *args, num_runs=self.num_benchmark_runs, warmup_runs=self.warmup_runs
        )

        backward_time = None
        if benchmark_backward:
            # Prepare args with gradient tracking for backward pass
            args_grad = []
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.dtype.is_floating_point:
                    args_grad.append(arg.clone().detach().requires_grad_(True))
                else:
                    args_grad.append(arg)

            backward_time = benchmark_function_backward(
                func,
                loss_fn,
                *args_grad,
                num_runs=self.num_benchmark_runs,
                warmup_runs=self.warmup_runs,
            )

        return forward_time, backward_time

    def run_validation(self) -> List[List[Any]]:
        """Run validation for all kernels and return results table"""
        print("=" * 80)
        print("CUDA KERNEL VALIDATION")
        print("=" * 80)
        print(
            f"Config: B={self.B}, N={self.N}, C={self.C}, dim={self.dim}, "
            f"eps={self.eps}, sigma={self.sigma}, \n"
            f"grid_size={self.grid_size}, boundary={self.boundary}, mode={self.mode}, "
            f"max_particles_per_block={self.max_particles_per_block}"
        )
        print(
            f"Device: {self.device}, Seed: {self.seed}, Runs: {self.num_benchmark_runs}, Warmup: {self.warmup_runs}"
        )
        print("=" * 80)

        kernels = self._define_kernels()
        results = []

        for kernel in kernels:
            forward_error, backward_error, _, _ = self.validate_kernel(kernel)

            forward_status = "✓" if forward_error < 1e-4 else "✗"
            backward_status = "N/A"
            if backward_error is not None:
                backward_status = "✓" if backward_error < 1e-4 else "✗"

            results.append(
                [
                    kernel.name,
                    f"{forward_error:.2e}",
                    forward_status,
                    f"{backward_error:.2e}" if backward_error is not None else "N/A",
                    backward_status,
                ]
            )

        return results

    def run_benchmarks(self) -> List[List[Any]]:
        """Run benchmarks for all kernels and return results table"""
        kernels = self._define_kernels()
        special_benchmarks = self._define_special_benchmarks()
        results = []

        # Benchmark regular kernels
        for kernel in [k for k in kernels if k.benchmark]:
            args = kernel.args_func()
            forward_time, backward_time = self.benchmark_kernel(
                kernel.name,
                kernel.cuda_func,
                args,
                benchmark_backward=kernel.has_backward,
            )

            results.append(
                [
                    kernel.name,
                    f"{forward_time:.3f}",
                    f"{backward_time:.3f}" if backward_time is not None else "N/A",
                ]
            )

        # Benchmark special cases
        for name, func, args in special_benchmarks:
            forward_time, _ = self.benchmark_kernel(
                name, func, args, benchmark_backward=False
            )
            results.append([name, f"{forward_time:.3f}", "N/A"])

        return results

    def print_results(
        self, validation_results: List[List[Any]], benchmark_results: List[List[Any]]
    ):
        """Print formatted results tables"""
        def format_table(data, headers):
            all_rows = [headers] + data
            col_widths = [max(len(str(item)) for item in col) for col in zip(*all_rows)]
            
            def format_row(row):
                return " | ".join(str(item).ljust(w) for item, w in zip(row, col_widths))
            
            separator = f"+-{'-+-'.join('-' * w for w in col_widths)}-+"
            header_separator = f"+={'=+='.join('=' * w for w in col_widths)}=+"
            
            lines = [separator, f"| {format_row(headers)} |", header_separator]
            for row in data:
                lines.append(f"| {format_row(row)} |")
                lines.append(separator)
            return "\n".join(lines)

        if validation_results:
            headers = ["Kernel", "Forward Error", "Fwd ✓", "Backward Error", "Bwd ✓"]
            print(format_table(validation_results, headers))

        if benchmark_results:
            headers = ["Kernel", "Forward (ms)", "Backward (ms)"]
            print(format_table(benchmark_results, headers))

    #         print("=" * 80)

    def run_all(self, validate=True):
        """Run complete validation and benchmark suite"""
        validation_results, benchmark_results = None, None
        if validate:
            validation_results = self.run_validation()
        benchmark_results = self.run_benchmarks()
        self.print_results(validation_results, benchmark_results)

        # Summary
        if validation_results:
            all_forward_passed = all(result[2] == "✓" for result in validation_results)
            all_backward_passed = all(
                result[4] == "✓" or result[4] == "N/A" for result in validation_results
            )
            all_passed = all_forward_passed and all_backward_passed

            print(
                f"\n{'✓' if all_passed else '✗'} Validation: "
                f"{'All tests passed!' if all_passed else 'Some tests failed!'}"
            )
            print(f"  Forward: {'✓ Passed' if all_forward_passed else '✗ Failed'}")
            print(f"  Backward: {'✓ Passed' if all_backward_passed else '✗ Failed'}")


def main():
    """Main entry point"""
    for boundary in ["clamped", "periodic"]:
        for mode in ["particle", "grid"]:
            print(f"\nTesting with boundary={boundary}, mode={mode}")


            validator = CUDAKernelValidator(
                grid_size=16,
                # boundary="clamped",
                boundary=boundary,
                
                # mode="grid",
                mode=mode,
                B=3,
                N=256,
                C=16,
                dim=2,
                # eps=0.1,
                eps=0.1,
                
                sigma=1.0,
                # sigma=0.2,
                
                max_particles_per_block=64,
                rand_type="rand",
                device="cuda:0",
                num_benchmark_runs=64,
                warmup_runs=16,
                seed=42,  # Set to None for non-reproducible random data
            )

            validator.run_all(True)


if __name__ == "__main__":
    main()
