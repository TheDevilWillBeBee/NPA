"""CPU checks by default; CUDA numerical and rollout checks require --gpu."""

import argparse
import gc
import importlib
import itertools
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np
from PIL import Image
import torch

from sphops.benchmark_data import build_cases, disk_positions, image_positions


class CPUPerceptionTests(unittest.TestCase):
    def test_benchmark_permutation_preserves_particle_values_and_cotangents(self):
        from sphops.benchmark import canonical_particles, binned_particles
        permutation = torch.tensor([[2, 0, 1]], dtype=torch.int32)
        values = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4).requires_grad_(True)
        binned, = binned_particles((values,), permutation)
        canonical, = canonical_particles((binned,), permutation)
        torch.testing.assert_close(canonical, values, rtol=0, atol=0)
        cotangent = torch.arange(12, dtype=torch.float32).reshape_as(values) / 10
        binned_cotangent, = binned_particles((cotangent,), permutation)
        actual, = torch.autograd.grad(binned, values, binned_cotangent)
        torch.testing.assert_close(actual, cotangent, rtol=0, atol=0)

    def test_package_and_standalone_imports_share_one_native_extension(self):
        from sphops import sph
        with patch.object(sys, "path", [str(Path(sph.__file__).parent), *sys.path]):
            standalone = importlib.import_module("sph")
        self.assertIs(standalone.sph_cuda, sph.sph_cuda)

    def test_composite_perception_packing_and_optional_features(self):
        from sphops import sph
        state = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
        position = torch.zeros(1, 3, 2)
        raw_gradient = torch.arange(48, dtype=torch.float32).reshape(1, 3, 8, 2) / 10
        density_gradient = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2) / 10
        snapshot = SimpleNamespace(grid=SimpleNamespace(eps=0.1), dim=2, mode="grid")
        eps0 = torch.tensor([0.2])
        with patch.object(sph, "density", return_value=torch.ones(1, 3)), \
             patch.object(sph, "blur", return_value=state * 2), \
             patch.object(sph, "gradient_hybrid", return_value=raw_gradient), \
             patch.object(sph, "density_gradient", return_value=density_gradient):
            for state_grad, density_grad, log_grad, log_density in itertools.product((False, True), repeat=4):
                output = sph.perception(
                    position, state, snapshot, eps0, state_grad=state_grad,
                    density_grad=density_grad, log_norm_grad=log_grad,
                    log_norm_density_grad=log_density, use_fused=False,
                )
                expected = [state, state * 2]
                if state_grad:
                    value = raw_gradient * 0.5
                    if log_grad:
                        value = torch.log(1 + value.norm(dim=-1, keepdim=True)) * torch.nn.functional.normalize(value, dim=-1)
                    expected.append(value.flatten(-2))
                if density_grad:
                    value = density_gradient * (0.5 ** 3) / 3
                    if log_density:
                        value = torch.log(1 + value.norm(dim=-1, keepdim=True)) * torch.nn.functional.normalize(value, dim=-1)
                    expected.append(value)
                torch.testing.assert_close(output, torch.cat(expected, -1), rtol=0, atol=0)

    def test_cpu_inputs_are_not_reported_as_fused_supported(self):
        from sphops import fused_perception_supported
        snapshot = SimpleNamespace(grid=SimpleNamespace(perception_forward=object()), mode="grid")
        self.assertFalse(fused_perception_supported(torch.zeros(1, 3, 2), torch.zeros(1, 3, 16), snapshot))

    def test_fused_channel_buckets_and_unsupported_sizes(self):
        from sphops.sph import _fused_perception_channel_count
        for channels in range(1, 33):
            expected = 4 if channels <= 4 else 8 if channels <= 8 else 16 if channels <= 16 else 32
            self.assertEqual(_fused_perception_channel_count(channels), expected)
        for channels in (-1, 0, 33, 64):
            self.assertIsNone(_fused_perception_channel_count(channels))

    def test_padded_perception_field_packing_and_autograd(self):
        from sphops import sph
        # Mock only the CUDA call: exercise real PyTorch padding, field slices,
        # concatenation, and their adjoints for every logical channel count.
        for dim, channels, density_grad in itertools.product((2, 3), range(1, 33), (False, True)):
            with self.subTest(dim=dim, channels=channels, density_grad=density_grad):
                state = torch.randn(1, 3, channels, requires_grad=True)
                position = torch.zeros(1, 3, dim)
                snapshot = SimpleNamespace(dim=dim, mode="grid")
                captured = {}
                def fake_native(x, padded, mass, snapshot, eps0, density_grad, *flags):
                    padded.retain_grad()
                    torch.testing.assert_close(mass, torch.ones_like(state[..., 0]), rtol=0, atol=0)
                    captured["state"] = padded
                    gradient = padded.unsqueeze(-1).expand(*padded.shape, dim) * 3
                    parts = [padded, padded * 2, gradient.flatten(-2)]
                    if density_grad:
                        parts.append(torch.arange(dim, dtype=padded.dtype).expand(1, 3, dim) + 10)
                    captured["output"] = torch.cat(parts, -1)
                    captured["output"].retain_grad()
                    return captured["output"]
                with patch.object(sph, "fused_perception_supported", return_value=True), \
                     patch.object(sph.FusedPerception, "apply", side_effect=fake_native):
                    output = sph.perception(position, state, snapshot, torch.tensor([0.1]),
                                            density_grad=density_grad)
                padded_channels = sph._fused_perception_channel_count(channels)
                self.assertEqual(captured["state"].shape[-1], padded_channels)
                torch.testing.assert_close(captured["state"][..., :channels], state, rtol=0, atol=0)
                self.assertEqual(int(torch.count_nonzero(captured["state"][..., channels:])), 0)
                parts = [state, state * 2, (state.unsqueeze(-1).expand(*state.shape, dim) * 3).flatten(-2)]
                if density_grad:
                    parts.append(torch.arange(dim, dtype=state.dtype).expand(1, 3, dim) + 10)
                expected = torch.cat(parts, -1)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                self.assertTrue(output.is_contiguous())
                if channels == padded_channels:
                    self.assertIs(output, captured["output"])
                cotangent = torch.randn_like(output)
                expected_grad, = torch.autograd.grad(expected, state, cotangent)
                output.backward(cotangent)
                torch.testing.assert_close(state.grad, expected_grad, rtol=1e-6, atol=1e-6)
                if channels != padded_channels:
                    self.assertEqual(int(torch.count_nonzero(captured["state"].grad[..., channels:])), 0)
                    native_grad = captured["output"].grad
                    for begin, end in ((channels, padded_channels),
                                       (padded_channels + channels, 2 * padded_channels),
                                       (2 * padded_channels + channels * dim, padded_channels * (2 + dim))):
                        self.assertEqual(int(torch.count_nonzero(native_grad[..., begin:end])), 0)

    def test_fixed_mass_validation_and_legacy_argument_propagation(self):
        from sphops import sph
        x, state = torch.zeros(1, 3, 2), torch.zeros(1, 3, 4)
        snapshot = SimpleNamespace(grid=SimpleNamespace(eps=0.1), dim=2, mode="grid")
        eps0 = torch.tensor([0.1])
        for mass in (torch.ones(1, 2), torch.ones(1, 3, dtype=torch.float64),
                     torch.ones(1, 3, requires_grad=True)):
            with self.subTest(shape=mass.shape, dtype=mass.dtype, requires_grad=mass.requires_grad):
                with self.assertRaises(ValueError):
                    sph.perception(x, state, snapshot, eps0, mass=mass)
        storage = torch.tensor([[[0.2, 9], [0.7, 9], [1.3, 9]]])
        mass = storage[..., :1]  # [B,N,1], noncontiguous: normalize once for native calls.
        with patch.object(sph, "density", return_value=torch.ones(1, 3)) as density, \
             patch.object(sph, "blur", return_value=state) as blur, \
             patch.object(sph, "gradient_hybrid", return_value=torch.zeros(1, 3, 4, 2)) as gradient, \
             patch.object(sph, "density_gradient", return_value=x) as density_gradient:
            sph.perception(x, state, snapshot, eps0, mass=mass, use_fused=False)
            for operation in (density, blur, gradient, density_gradient):
                actual_mass = operation.call_args.kwargs["mass"]
                self.assertTrue(actual_mass.is_contiguous())
                torch.testing.assert_close(actual_mass, mass.squeeze(-1), rtol=0, atol=0)

    def test_binning_keeps_fixed_mass_nondifferentiable_with_trainable_state(self):
        from sphops.sph import Permute
        def copy_forward(x, state, mass, permutation, out_x, out_state, out_mass):
            out_x.copy_(x)
            out_state.copy_(state)
            out_mass.copy_(mass)
        def copy_backward(gx, gs, gm, permutation, out_gx, out_gs, out_gm):
            out_gx.copy_(gx)
            out_gs.copy_(gs)
            out_gm.copy_(gm)
        grid = SimpleNamespace(bin_particles=copy_forward, backward_bin_particles=copy_backward)
        snapshot = SimpleNamespace(grid=grid, permutation=torch.arange(3).reshape(1, 3))
        for trainable_mass in (False, True):
            state = torch.randn(1, 3, 4, requires_grad=True)
            mass = torch.tensor([[0.2, 0.7, 1.3]], requires_grad=trainable_mass)
            _, state_bin, mass_bin = Permute.apply(torch.zeros(1, 3, 2), state, mass, snapshot)
            self.assertEqual(mass_bin.requires_grad, trainable_mass)
            (state_bin.sum() + mass_bin.sum()).backward()
            torch.testing.assert_close(state.grad, torch.ones_like(state))
            if trainable_mass:
                torch.testing.assert_close(mass.grad, torch.ones_like(mass))
            else:
                self.assertIsNone(mass.grad)

    def test_fused_default_and_opt_out_preserve_checkpoint_structure(self):
        from models.npa import NPA
        legacy = NPA(hidden_dims=32, fused_perception=False)
        candidate = NPA(hidden_dims=32)
        candidate.load_state_dict(legacy.state_dict(), strict=True)
        self.assertFalse(legacy.fused_perception)
        self.assertTrue(candidate.fused_perception)
        self.assertEqual(list(candidate.state_dict()), list(legacy.state_dict()))

    def test_cached_state_adjoint_preserves_straight_through_correction(self):
        # Independent autograd reference for the complete state operator, with a
        # deliberately non-identity moment correction and non-unit radius scale.
        for dim in (2, 3):
            generator = torch.Generator().manual_seed(42 + dim)
            N, C = 9, 5
            state = torch.randn(N, C, dtype=torch.float64, generator=generator, requires_grad=True)
            volume = torch.rand(N, dtype=torch.float64, generator=generator) + 0.1
            weight = torch.rand(N, N, dtype=torch.float64, generator=generator)
            weight = (weight + weight.T) / 2
            kernel = torch.randn(N, N, dim, dtype=torch.float64, generator=generator)
            kernel = kernel - kernel.transpose(0, 1)
            matrix = torch.randn(N, dim, dim, dtype=torch.float64, generator=generator)
            blur = torch.einsum("ij,j,jc->ic", weight, volume, state)
            raw = torch.einsum(
                "ijd,j,ijc->icd", kernel, volume, state[None] - state[:, None],
            )
            corrected = raw + (raw @ matrix - raw).detach()
            scaled = corrected * 0.7
            normalized = torch.log(1 + scaled.norm(dim=-1, keepdim=True)) * torch.nn.functional.normalize(scaled, dim=-1)
            direct_grad = torch.randn(state.shape, dtype=state.dtype, generator=generator)
            blur_grad = torch.randn(blur.shape, dtype=state.dtype, generator=generator)
            norm_grad = torch.randn(normalized.shape, dtype=state.dtype, generator=generator)
            loss = (state * direct_grad).sum() + (blur * blur_grad).sum() + (normalized * norm_grad).sum()
            expected, raw_grad = torch.autograd.grad(loss, (state, raw))
            cached_sum = torch.einsum("ijd,j->id", kernel, volume)
            actual = direct_grad - torch.einsum("icd,id->ic", raw_grad, cached_sum)
            actual += volume[:, None] * (
                torch.einsum("ij,jc->ic", weight, blur_grad)
                - torch.einsum("ijd,jcd->ic", kernel, raw_grad)
            )
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
            wrong = raw_grad @ matrix.transpose(-1, -2)
            self.assertGreater(float((wrong - raw_grad).abs().max()), 0.01)

    def test_normalization_jacobian_including_zero_and_clamped_norms(self):
        generator = torch.Generator().manual_seed(17)
        vectors = torch.randn(8, 3, dtype=torch.float64, generator=generator)
        vectors[0] = 0
        vectors[1] *= 1e-14
        vectors[2] *= 1e-10
        vectors.requires_grad_(True)
        cotangent = torch.randn(vectors.shape, dtype=vectors.dtype, generator=generator)
        normalized = torch.log(1 + vectors.norm(dim=-1, keepdim=True)) * torch.nn.functional.normalize(vectors, dim=-1)
        expected, = torch.autograd.grad(normalized, vectors, cotangent)
        norm = vectors.detach().norm(dim=-1, keepdim=True)
        unit = vectors.detach() / norm.clamp_min(torch.finfo(norm.dtype).tiny)
        factor = torch.log(1 + norm) / norm.clamp_min(1e-12)
        radial = torch.where(norm > 1e-12, 1 / (1 + norm) - factor, norm / (1e-12 * (1 + norm)))
        actual = factor * cotangent + radial * unit * (unit * cotangent).sum(-1, keepdim=True)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)

    def test_seed_fixtures_are_deterministic_and_match_training_domains(self):
        manifest = Path(__file__).with_name("benchmark_targets.yaml")
        kwargs = dict(batch_size=2, num_particles=128, seed=42, selected=["growing_seed", "texture_seed"])
        first = build_cases(manifest, **kwargs)
        second = build_cases(manifest, **kwargs)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a.positions, b.positions)
            self.assertEqual(a.metadata()["positions_sha256"], b.metadata()["positions_sha256"])
        disk, texture = first
        self.assertLessEqual(float(np.linalg.norm(disk.positions, axis=-1).max()), 0.2 + 1e-7)
        self.assertEqual(disk.boundary, "clamped")
        self.assertEqual(texture.grid_size, (10, 10))
        self.assertTrue(np.all(np.abs(texture.positions) <= 1))

    def test_alpha_sampling_is_unique_and_rejects_empty_targets(self):
        rgba = np.zeros((16, 16, 4), dtype=np.uint8)
        rgba[3:13, 4:12, 3] = 255
        image = Image.fromarray(rgba)
        a, metadata = image_positions(image, np.random.default_rng(42), 2, 128)
        b, _ = image_positions(image, np.random.default_rng(42), 2, 128)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(metadata["replacement"])
        for batch in a:
            self.assertEqual(len(np.unique(batch, axis=0)), 128)
        with self.assertRaisesRegex(ValueError, "no pixels"):
            image_positions(Image.fromarray(np.zeros_like(rgba)), np.random.default_rng(42), 1, 16)

    def test_manifest_reads_multiple_targets_and_reports_missing_assets(self):
        with tempfile.TemporaryDirectory(prefix="npa-perception-tests-") as directory:
            root = Path(directory)
            Image.new("RGBA", (16, 16), (255, 0, 0, 255)).save(root / "target.png")
            manifest = root / "targets.yaml"
            # Generated fixtures stay inside the test's temporary directory.
            manifest.write_text("asset_root: .\nemoji_targets:\n  - name: test_emoji\n    path: target.png\ntexture_targets:\n  - name: test_texture\n    path: target.png\n")
            cases = build_cases(manifest, batch_size=1, num_particles=32,
                                selected=["test_emoji", "test_texture"])
            self.assertEqual([case.family for case in cases], ["emoji_target", "texture_target"])
            self.assertEqual(cases[0].positions.dtype, np.float32)
            with self.assertRaisesRegex(ValueError, "Unknown cases"):
                build_cases(manifest, selected=["unknown"])
            (root / "target.png").unlink()
            with self.assertRaisesRegex(ValueError, "Missing asset"):
                build_cases(manifest, selected=["test_emoji"])


class CUDAPerceptionTests(unittest.TestCase):
    instrumented = False

    @classmethod
    def setUpClass(cls):
        if os.environ.get("NPA_RUN_CUDA_TESTS") != "1":
            raise unittest.SkipTest("CUDA checks require --gpu or NPA_RUN_CUDA_TESTS=1")
        from sphops.sph import sph_cuda
        if not hasattr(sph_cuda.HashGrid2DBox, "perception_forward"):
            raise RuntimeError("Rebuild the CUDA extension or select a new build with NPA_SPH_EXTENSION_PATH before CUDA checks")

    def reference(self, x, s, grid, eps0, mass=None, density_grad=True,
                  log_norm_grad=True, log_norm_density_grad=True):
        from sphops import torch_sph
        x = x[..., :grid.dim]
        periodic = grid.boundary == "periodic"
        length = [size * grid.eps for size in grid.grid_size] if periodic else None
        rho = torch_sph.density(x, grid.eps, mass, periodic, length)
        blur = torch_sph.blur(x, rho, s, grid.eps, mass, periodic, length)
        with torch.no_grad():
            matrix = torch_sph.moment_matrix(x, rho, grid.eps, mass, periodic, length)
            inverse, _ = torch.linalg.inv_ex(matrix)
            singular = torch.linalg.det(matrix).abs() < 1e-3
            inverse = torch.where(singular[..., None, None], torch.eye(grid.dim, device=x.device), inverse)
        raw = torch_sph.gradient(x, rho, s, grid.eps, mass, periodic, length)
        gradient = (raw + (raw @ inverse - raw).detach()) * (grid.eps / eps0)
        if log_norm_grad:
            gradient = torch.log(1 + gradient.norm(dim=-1, keepdim=True)) * torch.nn.functional.normalize(gradient, dim=-1)
        parts = [s, blur, gradient.flatten(-2)]
        if density_grad:
            grad_rho = torch_sph.density_gradient(x, grid.eps, mass, periodic, length)
            grad_rho = grad_rho * (grid.eps / eps0) ** (grid.dim + 1) / s.shape[1]
            if log_norm_density_grad:
                grad_rho = torch.log(1 + grad_rho.norm(dim=-1, keepdim=True)) * torch.nn.functional.normalize(grad_rho, dim=-1)
            parts.append(grad_rho)
        return torch.cat(parts, -1)

    def fixture(self, dim=2, channels=16, boundary="clamped", cap=32, size=8, dense=True):
        from sphops import HashGrid, bin_particles
        generator = torch.Generator(device="cuda").manual_seed(42)
        x = torch.rand(2, 96, dim, generator=generator, device="cuda")
        x = (x - 0.5) * (0.3 if dense else size * 0.1)
        if dim == 3:
            x = torch.cat([x, torch.zeros_like(x[..., :1])], -1)
        state = torch.randn(2, 96, channels, generator=generator, device="cuda") * 0.1
        grid = HashGrid(dim=dim, boundary=boundary, mode="grid", grid_size=[size] * dim,
                        eps=0.1, num_particles=96, batch_size=2, max_particles_per_block=cap)
        snapshot = grid.instantiate(x)
        x, state = bin_particles(x, state, snapshot)
        torch.cuda.synchronize()
        return x, state, grid, snapshot

    def compare(self, x, state, grid, snapshot, **flags):
        from sphops import perception
        from sphops.benchmark import cached_state_adjoint_reference, fused_perception_context
        eps0 = torch.tensor([grid.eps / 0.7], device="cuda")
        state = state.detach().requires_grad_(True)
        reference_state = state.detach().clone().requires_grad_(True)
        output = perception(x, state, snapshot, eps0, **flags)
        native_context = fused_perception_context(output)
        self.assertEqual(native_context.state_shape[-1], next(
            count for count in (4, 8, 16, 32) if state.shape[-1] <= count))
        expected = self.reference(x, reference_state, grid, eps0, **flags)
        torch.testing.assert_close(output, expected, rtol=3e-4, atol=4e-5)
        generator = torch.Generator(device=state.device).manual_seed(71)
        cotangent = torch.randn(output.shape, device=state.device, generator=generator)
        cached_reference_grad = cached_state_adjoint_reference(
            torch, x, state, snapshot, eps0, output, cotangent, mass=flags.get("mass"))
        actual_grad, = torch.autograd.grad(output, state, cotangent)
        expected_grad, = torch.autograd.grad(expected, reference_state, cotangent)
        # Strictly verify the derivative independently at identical cached
        # normalization values, including padded and nonunit-mass paths.
        torch.testing.assert_close(actual_grad, cached_reference_grad, rtol=4e-4, atol=5e-5)
        try:
            torch.testing.assert_close(actual_grad, expected_grad, rtol=4e-4, atol=5e-5)
        except AssertionError:
            if not native_context.log_norm_grad:
                raise
            # Float32's rounded (1+norm) can give different Jacobians for
            # roundoff-close forward values. Do not relax the cached-point
            # check; additionally require normwise agreement with the dense
            # independent SPH forward point, as the full-size benchmark does.
            difference = (actual_grad - expected_grad).norm() / expected_grad.norm().clamp_min(1e-12)
            self.assertLessEqual(float(difference), 1e-5)

    def test_forward_and_backward_dimensions_boundaries_channels_and_block_caps(self):
        for dim, boundary, channels, cap in itertools.product((2, 3), ("clamped", "periodic"), (1, 4, 5, 8, 16, 17, 32), (16, 32, 64)):
            with self.subTest(dim=dim, boundary=boundary, channels=channels, cap=cap):
                self.compare(*self.fixture(dim, channels, boundary, cap))

    def test_nonunit_mass_dimensions_boundaries_channels_and_block_caps(self):
        for dim, boundary, channels, cap in itertools.product((2, 3), ("clamped", "periodic"),
                                                             (1, 4, 5, 8, 16, 17, 32), (16, 32, 64)):
            with self.subTest(dim=dim, boundary=boundary, channels=channels, cap=cap):
                fixture = self.fixture(dim, channels, boundary, cap)
                generator = torch.Generator(device="cuda").manual_seed(93)
                mass = 0.2 + 2.5 * torch.rand(2, 96, generator=generator, device="cuda")
                self.compare(*fixture, mass=mass)

    def test_nonunit_mass_flags_layouts_and_global_scaling(self):
        from sphops import perception
        for dim, channels in itertools.product((2, 3), (4, 5, 17)):
            fixture = self.fixture(dim=dim, channels=channels)
            x, state, grid, snapshot = fixture
            generator = torch.Generator(device="cuda").manual_seed(94)
            mass = 0.3 + torch.rand(2, 96, 2, generator=generator, device="cuda")
            mass = mass[..., :1]  # [B,N,1] and noncontiguous.
            for density_grad, log_norm_grad, log_norm_density_grad in itertools.product((False, True), repeat=3):
                with self.subTest(dim=dim, channels=channels, density_grad=density_grad,
                                  log_norm_grad=log_norm_grad, log_norm_density_grad=log_norm_density_grad):
                    self.compare(*fixture, mass=mass, density_grad=density_grad,
                                 log_norm_grad=log_norm_grad, log_norm_density_grad=log_norm_density_grad)
            for constant in (1.0 / 96, 0.1, 2.0):
                with self.subTest(dim=dim, channels=channels, constant_mass=constant):
                    self.compare(*fixture, mass=torch.full((2, 96), constant, device="cuda"))
            # Uniform rescaling leaves m/rho and all state features unchanged,
            # but scales the raw density-gradient feature before log normalization.
            with torch.no_grad():
                eps0 = torch.tensor([grid.eps], device="cuda")
                first = perception(x, state, snapshot, eps0, mass=mass, log_norm_density_grad=False)
                scaled = perception(x, state, snapshot, eps0, mass=mass * 3, log_norm_density_grad=False)
                split = channels * (2 + dim)
                torch.testing.assert_close(first[..., :split], scaled[..., :split], rtol=3e-4, atol=4e-5)
                torch.testing.assert_close(first[..., split:] * 3, scaled[..., split:], rtol=3e-4, atol=4e-5)

    def test_normalization_flags_and_zero_state(self):
        for channels in (1, 4, 5, 16, 17):
            fixture = self.fixture(channels=channels)
            for density_grad, log_norm_grad, log_norm_density_grad in itertools.product((False, True), repeat=3):
                with self.subTest(channels=channels, density_grad=density_grad, log_norm_grad=log_norm_grad,
                                  log_norm_density_grad=log_norm_density_grad):
                    self.compare(*fixture, density_grad=density_grad, log_norm_grad=log_norm_grad,
                                 log_norm_density_grad=log_norm_density_grad)
            x, state, grid, snapshot = fixture
            self.compare(x, torch.zeros_like(state), grid, snapshot)
            self.compare(x, torch.ones_like(state), grid, snapshot)

    def test_padded_inference_and_oversize_fallback(self):
        from sphops import fused_perception_supported, perception
        for channels in (1, 4, 5, 17, 33):
            with self.subTest(channels=channels):
                x, state, grid, snapshot = self.fixture(channels=channels)
                self.assertEqual(fused_perception_supported(x, state, snapshot), channels <= 32)
                eps0 = torch.tensor([grid.eps], device="cuda")
                with torch.no_grad():
                    actual = perception(x, state, snapshot, eps0)
                    expected = perception(x, state, snapshot, eps0, use_fused=False)
                self.assertIsNone(actual.grad_fn)
                torch.testing.assert_close(actual, expected, rtol=3e-4, atol=4e-5)

    def test_tiny_state_gradients_preserve_normalization_backward_accuracy(self):
        for channels in (1, 4, 5, 16, 17):
            x, state, grid, snapshot = self.fixture(channels=channels)
            for scale in (1e-4, 1e-5, 1e-7):
                with self.subTest(channels=channels, state_scale=scale):
                    self.compare(x, state * scale, grid, snapshot)

    def test_backward_releases_snapshot_and_grid_owners(self):
        if self.instrumented:
            raise unittest.SkipTest("CUDA instrumentation retains Python allocation/call frames; checked without instrumentation")
        from sphops import perception
        def iteration(channels):
            x, state, grid, snapshot = self.fixture(channels=channels)
            state.requires_grad_(True)
            output = perception(x, state, snapshot, torch.tensor([grid.eps], device="cuda"))
            torch.autograd.grad(output, state, torch.ones_like(output))
            return weakref.ref(grid), weakref.ref(snapshot)
        references = [iteration(channels) for channels in (1, 4, 5, 16, 17) for _ in range(4)]
        torch.cuda.synchronize()
        gc.collect()
        for grid_ref, snapshot_ref in references:
            self.assertIsNone(grid_ref())
            self.assertIsNone(snapshot_ref())

    def test_periodic_seams_outside_box_and_non_power_of_two_grid(self):
        from sphops import bin_particles
        x, state, grid, _ = self.fixture(boundary="periodic", size=10, dense=False)
        x = x.clone()
        x[:, :2, 0] = torch.tensor([-0.499, 0.499], device="cuda")
        x[:, :2, 1] = 0
        x[:, 2:10] += torch.tensor([2.0, -3.0], device="cuda")
        snapshot = grid.instantiate(x)
        x, state = bin_particles(x, state, snapshot)
        self.compare(x, state, grid, snapshot)

    def test_duplicate_and_isolated_positions_with_singular_moment_matrices(self):
        from sphops import HashGrid, bin_particles
        x = torch.tensor([[[-0.4, -0.4], [-0.4, -0.4], [0.4, 0.4]]], device="cuda")
        state = torch.randn(1, 3, 16, device="cuda")
        grid = HashGrid(dim=2, boundary="clamped", mode="grid", grid_size=[10, 10],
                        eps=0.1, num_particles=3, batch_size=1, max_particles_per_block=32)
        snapshot = grid.instantiate(x)
        x, state = bin_particles(x, state, snapshot)
        self.compare(x, state, grid, snapshot)

    def test_saved_snapshot_survives_later_grid_initializations(self):
        from sphops import HashGrid, bin_particles, blur, density, perception
        axis = (torch.arange(16, device="cuda", dtype=torch.float32) - 8 + 0.5) * 0.1
        sparse = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), -1).reshape(1, 256, 2).repeat_interleave(2, 1)
        state = torch.randn(1, 512, 16, device="cuda", requires_grad=True)
        grid = HashGrid(dim=2, boundary="clamped", mode="grid", grid_size=[16, 16],
                        eps=0.1, num_particles=512, batch_size=1, max_particles_per_block=32)
        snapshot = grid.instantiate(sparse)
        x, s = bin_particles(sparse, state, snapshot)
        rho = density(x, snapshot)
        blur_output = blur(x, rho, s, snapshot)
        eps0 = torch.tensor([0.1], device="cuda")
        fused = perception(x, s, snapshot, eps0)
        blur_cotangent, cotangent = torch.randn_like(blur_output), torch.randn_like(fused)
        before_blur, = torch.autograd.grad(blur_output, state, blur_cotangent, retain_graph=True)
        before, = torch.autograd.grad(fused, state, cotangent, retain_graph=True)
        grid.instantiate(torch.full_like(sparse, 0.02))
        after_blur, = torch.autograd.grad(blur_output, state, blur_cotangent, retain_graph=True)
        after, = torch.autograd.grad(fused, state, cotangent)
        torch.testing.assert_close(after_blur, before_blur, rtol=0, atol=0)
        torch.testing.assert_close(after, before, rtol=0, atol=0)
        torch.testing.assert_close(density(x, snapshot), rho, rtol=0, atol=0)

    def test_fallback_retains_full_position_and_density_backward(self):
        from sphops import fused_perception_supported, perception
        x, state, grid, snapshot = self.fixture(channels=12)
        x = x.detach().requires_grad_(True)
        state = state.detach().requires_grad_(True)
        eps0 = torch.tensor([grid.eps], device="cuda")
        mass = torch.linspace(0.3, 1.7, 96, device="cuda").expand(2, 96).contiguous()
        self.assertFalse(fused_perception_supported(x, state, snapshot, mass))
        actual = perception(x, state, snapshot, eps0, mass=mass)
        legacy = perception(x, state, snapshot, eps0, mass=mass, use_fused=False)
        expected = self.reference(x, state, grid, eps0, mass=mass)
        torch.testing.assert_close(actual, legacy, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=4e-5)
        for seed in range(8):
            with self.subTest(cotangent_seed=seed):
                generator = torch.Generator(device="cuda").manual_seed(seed)
                cotangent = torch.randn(actual.shape, device="cuda", generator=generator)
                actual_grads = torch.autograd.grad(actual, (x, state), cotangent, retain_graph=True)
                legacy_grads = torch.autograd.grad(legacy, (x, state), cotangent, retain_graph=True)
                expected_grads = torch.autograd.grad(expected, (x, state), cotangent, retain_graph=True)
                for index, (a, old, b) in enumerate(zip(actual_grads, legacy_grads, expected_grads)):
                    # Exact legacy dispatch, allowing its atomic accumulation
                    # ordering; keep a strict pointwise state-adjoint check.
                    torch.testing.assert_close(a, old, rtol=5e-4, atol=1e-4 if index == 0 else 5e-5)
                    if index == 1:
                        torch.testing.assert_close(a, b, rtol=5e-4, atol=5e-5)
                    else:
                        # Position gradients have large canceling terms. The
                        # dense float32 reference differs by <5e-7 relative L2
                        # on these eight cotangents, but a near-zero component
                        # can fail an absolute 1e-4 pointwise test.
                        relative_error = (a - b).norm() / b.norm().clamp_min(1e-12)
                        self.assertLessEqual(float(relative_error), 3e-6)

    def test_multi_step_npa_parameter_and_state_gradients(self):
        for channels in (1, 4, 5, 16, 17):
            with self.subTest(channels=channels):
                self.compare_multi_step_npa(channels)

    def test_fixed_nonunit_mass_multi_step_npa_gradients_and_binning(self):
        for channels in (4, 5, 8, 16, 17, 32):
            with self.subTest(channels=channels):
                self.compare_multi_step_npa(channels, nonunit_mass=True)

    def compare_multi_step_npa(self, channels, nonunit_mass=False):
        from models.npa import NPA, step_euler
        from sphops import HashGrid
        torch.manual_seed(42)
        legacy = NPA(state_dims=channels, hidden_dims=32, fused_perception=False).cuda()
        fused = NPA(state_dims=channels, hidden_dims=32).cuda()
        self.assertTrue(fused.fused_perception)
        fused.load_state_dict(legacy.state_dict())
        initial_x = torch.randn(1, 96, 2, device="cuda") * 0.1
        initial_s = torch.randn(1, 96, channels, device="cuda") * 0.1
        mass = 0.3 + torch.rand(1, 96, device="cuda") if nonunit_mass else None
        results = []
        for model in (legacy, fused):
            grid = HashGrid(dim=2, boundary="clamped", mode="grid", grid_size=[16, 16],
                            eps=0.1, num_particles=96, batch_size=1, max_particles_per_block=32)
            state = initial_s.clone().requires_grad_(True)
            x, s = initial_x, state
            for _ in range(4):
                if mass is None:
                    x, s, _ = step_euler(model, x, s, grid, update_prob=1.0)
                else:
                    dx, ds, x_bin, s_bin = model(x, s, grid, mass=mass)
                    x, s = x_bin + dx, s_bin + ds
                # unbin_particles/step_euler_trace are visualization helpers
                # without autograd. Use the training step and differentiable
                # gathers to compare the two rollouts in original particle order.
                permutation = model.last_snapshot.permutation.long().unsqueeze(-1)
                x = x.gather(1, permutation.expand_as(x))
                s = s.gather(1, permutation.expand_as(s))
            loss = x.square().sum() + s.square().sum()
            gradients = torch.autograd.grad(loss, (state, *model.parameters()))
            results.append((x, s, gradients))
        for a, b in zip(results[0][:2], results[1][:2]):
            torch.testing.assert_close(a, b, rtol=5e-4, atol=5e-5)
        for a, b in zip(results[0][2], results[1][2]):
            torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", help="also run CUDA numerical and rollout checks")
    parser.add_argument("--instrumented", action="store_true",
                        help="skip Python object-lifetime assertions under CUDA instrumentation")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CPUPerceptionTests)
    if args.gpu:
        os.environ["NPA_RUN_CUDA_TESTS"] = "1"
        CUDAPerceptionTests.instrumented = args.instrumented
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(CUDAPerceptionTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if args.gpu:
        torch.cuda.synchronize()
    # Release autograd/context cycles before nanobind's interpreter-shutdown
    # diagnostics, including under Compute Sanitizer.
    gc.collect()
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
