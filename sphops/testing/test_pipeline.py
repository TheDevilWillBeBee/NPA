"""Focused tests for batched VGG features, batched splatting, and the optional compiled NPA update."""

import argparse
import gc
import itertools
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


class TexturePipelineTests(unittest.TestCase):
    def make_loss(self, *, device="cpu"):
        from losses.texture_loss import TextureLoss
        # No pretrained-weight download or native renderer is needed to check
        # batching and preservation of image and VGG parameter adjoints.
        loss = TextureLoss.__new__(TextureLoss)
        torch.nn.Module.__init__(loss)
        loss.style_layers = [1, 3]
        loss.vgg = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU(),
                                      torch.nn.Conv2d(4, 5, 3, padding=1), torch.nn.ReLU()).to(device)
        return loss

    def check_image_adjoint(self, device):
        torch.manual_seed(41)
        reference = self.make_loss(device=device)
        batched = self.make_loss(device=device)
        batched.load_state_dict(reference.state_dict())
        images = [torch.rand(2, 3, 16, 16, device=device), torch.rand(2, 3, 16, 16, device=device)]
        results = []
        for module, batch in ((reference, False), (batched, True)):
            inputs = [image.clone().requires_grad_() for image in images]
            features = (module.get_image_features(inputs) if batch
                        else [module.get_vgg_features(image) for image in inputs])
            value = sum((i + 1) * (j + 1) * feature.square().mean()
                        for i, layers in enumerate(features) for j, feature in enumerate(layers))
            value.backward()
            results.append((features, [image.grad for image in inputs],
                            [parameter.grad for parameter in module.parameters()]))
        for actual in results[1:]:
            for expected_layers, actual_layers in zip(results[0][0], actual[0]):
                for expected, value in zip(expected_layers, actual_layers):
                    torch.testing.assert_close(value, expected, rtol=2e-5, atol=2e-6)
            for value, expected in zip(actual[1], results[0][1]):
                torch.testing.assert_close(value, expected, rtol=3e-5, atol=2e-6)
            for value, expected in zip(actual[2], results[0][2]):
                torch.testing.assert_close(value, expected, rtol=3e-5, atol=2e-6)
        self.assertTrue(all(p.grad is not None for p in reference.parameters()))
        self.assertTrue(all(p.requires_grad and p.grad is not None for p in batched.parameters()))
        self.assertEqual(set(reference.state_dict()), set(batched.state_dict()))

    def test_batching_preserves_image_and_vgg_parameter_gradients(self):
        self.check_image_adjoint("cpu")

    def test_constructor_keeps_vgg_trainable_and_target_features_detached(self):
        from losses.texture_loss import TextureLoss
        vgg = self.make_loss().vgg

        def initialize_splat(loss, **kwargs):
            torch.nn.Module.__init__(loss)
            loss.target_color = torch.rand(1, 3, 8, 8)
            loss.target_density = torch.ones(1, 1, 8, 8)

        with patch("losses.texture_loss.SplatLoss.__init__", initialize_splat), \
                patch("losses.texture_loss.torchvision.models.vgg16",
                      return_value=SimpleNamespace(features=vgg)):
            loss = TextureLoss(target={}, style_layers=[1, 3], density_as_texture=True)
        self.assertTrue(loss.vgg.training)
        self.assertTrue(all(p.requires_grad for p in loss.vgg.parameters()))
        self.assertEqual(len(loss.target_features), 2)
        self.assertTrue(all(not feature.requires_grad
                            for layers in loss.target_features for feature in layers))
        self.assertEqual(dict(loss.named_buffers()), {})

    def test_different_image_batch_sizes_and_single_image(self):
        loss = self.make_loss()
        images = [torch.rand(2, 3, 8, 8), torch.rand(1, 3, 8, 8)]
        batched = loss.get_image_features(images)
        separate = [loss.get_vgg_features(image) for image in images]
        for actual, expected in zip(batched, separate):
            for value, reference in zip(actual, expected):
                torch.testing.assert_close(value, reference)
        self.assertEqual(len(loss.get_image_features(images[:1])), 1)


class CPUUpdateTests(unittest.TestCase):
    def test_tensor_only_update_matches_eager_including_noise_and_padding(self):
        from models.npa import NPA, step_euler
        for dim, intermediate, probability, noise_std in itertools.product(
                (2, 3), (False, True), (0.5, 1.0), (0.0, 0.02)):
            with self.subTest(dim=dim, intermediate=intermediate, probability=probability, noise=noise_std):
                torch.manual_seed(8)
                model = NPA(spatial_dims=dim, state_dims=5, hidden_dims=16)
                x = torch.randn(2, 7, 2 if dim == 2 else 4, requires_grad=True)
                state = torch.randn(2, 7, 5, requires_grad=True)
                z = torch.randn(2, 7, model.perception_dim, requires_grad=True)
                snapshot = object()
                grid = type("Grid", (), {"eps": 0.1})()
                # Keep the actual tensor-only function but avoid runtime
                # compilation in CPU tests; CUDA checks exercise Inductor.
                model._compiled_euler = model._euler_from_perception
                with patch.object(model, "perceive", return_value=(z, x, state, snapshot)):
                    torch.manual_seed(31)
                    expected = step_euler(model, x, state, grid, dt=0.3,
                        update_prob=probability, noise_std=noise_std, intermediate=intermediate)
                    model.compile_step = True
                    torch.manual_seed(31)
                    actual = step_euler(model, x, state, grid, dt=0.3,
                        update_prob=probability, noise_std=noise_std, intermediate=intermediate)
                for value, reference in zip(actual[:2], expected[:2]):
                    torch.testing.assert_close(value, reference, rtol=0, atol=0)
                self.assertEqual(actual[2] is None, expected[2] is None)
                if intermediate:
                    torch.testing.assert_close(actual[2]["dx_norm"], expected[2]["dx_norm"], rtol=0, atol=0)
                inputs = (x, state, z, *model.parameters())
                grads = []
                for output in (expected, actual):
                    loss = sum(value.square().sum() for value in output[:2])
                    if intermediate:
                        loss += output[2]["dx_norm"]
                    grads.append(torch.autograd.grad(loss, inputs, retain_graph=True))
                for value, reference in zip(grads[1], grads[0]):
                    torch.testing.assert_close(value, reference, rtol=0, atol=0)


class SplattingTests(unittest.TestCase):
    """Batched splatting against the per-sample reference implementation."""

    def compare(self, device, exact):
        from utils.splatting import splat_particles, splat_particles_batch
        generator = torch.Generator(device=device).manual_seed(5)
        for y_up, normalize_color in itertools.product((False, True), repeat=2):
            with self.subTest(device=device, y_up=y_up, normalize_color=normalize_color):
                kwargs = dict(img_size=48, sigma=1.3, lo=-1.0, hi=1.0, normalize_color=normalize_color,
                              y_up=y_up, pixel_size=2 / 40)
                # Spread beyond [-1, 1] so many kernel taps fall outside the image.
                x = (torch.randn(3, 200, 2, device=device, generator=generator) * 0.7).requires_grad_(True)
                color = torch.rand(3, 200, 5, device=device, generator=generator).requires_grad_(True)
                x[0, 0].data.fill_(9.0)  # One particle entirely outside the image.
                actual = splat_particles_batch(x, color, **kwargs)
                expected = torch.stack([splat_particles(x[b], color[b], **kwargs) for b in range(3)])
                cotangent = torch.randn(actual.shape, device=device, generator=generator)
                (x_grad, color_grad), (x_expected, color_expected) = (
                    torch.autograd.grad(output, (x, color), cotangent) for output in (actual, expected))
                # Same per-tap values; CUDA scatter_add adds them atomically (any order).
                if exact:
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                else:
                    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-6 * float(expected.abs().max()))
                # Color adjoints gather identical tap weights.
                torch.testing.assert_close(color_grad, color_expected, rtol=0, atol=0)
                # Separable tap distances reduce position adjoints in a different
                # order (P row sums of P taps instead of P*P taps): roundoff only.
                difference = x_grad - x_expected
                self.assertLessEqual(float(difference.norm() / x_expected.norm()), 1e-6)
                self.assertLessEqual(float(difference.abs().max()), 2e-6 * float(x_expected.abs().max()))

    def test_cpu_batched_splatting_matches_per_sample(self):
        # CPU scatter_add is sequential: images are exactly the per-sample ones.
        self.compare("cpu", exact=True)

    def test_off_image_nonfinite_colors_do_not_reach_the_image(self):
        from utils.splatting import splat_particles_batch
        x = torch.zeros(2, 3, 2)
        x[:, 0] = 7.0
        color = torch.rand(2, 3, 3)
        color[:, 0] = float("nan")
        image = splat_particles_batch(x, color, img_size=16, sigma=1.0, pixel_size=2 / 16)
        self.assertTrue(bool(torch.isfinite(image).all()))


class CUDAPipelineTests(unittest.TestCase):
    instrumented = False

    def test_cuda_batched_splatting_matches_per_sample(self):
        # CUDA scatter_add uses atomics, so summation order is not fixed.
        SplattingTests().compare("cuda", exact=False)

    def test_pretrained_vgg_batch_image_and_parameter_adjoints(self):
        import torchvision.models
        module = TexturePipelineTests().make_loss(device="cuda")
        module.vgg = torchvision.models.vgg16(weights="IMAGENET1K_V1").features.cuda()
        module.style_layers = [1, 6, 11, 18, 25]
        images = [torch.rand(2, 3, 64, 64, device="cuda") for _ in range(2)]
        # Full float32 isolates batching correctness from different TF32 cuDNN
        # algorithms. Production precision settings remain unchanged.
        with torch.backends.cudnn.flags(enabled=True, benchmark=False,
                deterministic=False, allow_tf32=False):
            results = []
            for batch in (False, True):
                module.vgg.zero_grad(set_to_none=True)
                inputs = [image.clone().requires_grad_() for image in images]
                features = (module.get_image_features(inputs) if batch
                            else [module.get_vgg_features(image) for image in inputs])
                value = sum(feature.square().mean() for layers in features for feature in layers)
                value.backward()
                parameters = tuple(module.vgg[:max(module.style_layers) + 1].parameters())
                self.assertTrue(all(p.requires_grad and p.grad is not None for p in parameters))
                results.append((value.detach(), [image.grad for image in inputs],
                                [p.grad.clone() for p in parameters]))
            torch.testing.assert_close(results[1][0], results[0][0], rtol=1e-5, atol=1e-5)
            for actual, expected in zip(results[1][1] + results[1][2],
                                        results[0][1] + results[0][2]):
                relative = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
                self.assertLess(float(relative), 1e-4)
                self.assertTrue(bool(torch.isfinite(actual).all()))

    def test_real_compiled_update_preserves_rng_and_parameter_gradients(self):
        if self.instrumented:
            self.skipTest("Inductor runtime compilation is checked without CUDA instrumentation")
        from models.npa import NPA, step_euler
        from sphops import HashGrid
        torch.manual_seed(4)
        model = NPA(state_dims=5, hidden_dims=32).cuda()
        grid = HashGrid(boundary="clamped", grid_size=[16, 16], eps=0.1, batch_size=1, num_particles=96)
        x = torch.randn(1, 96, 2, device="cuda") * 0.1
        state = torch.randn(1, 96, 5, device="cuda") * 0.1
        # One fixed snapshot avoids comparing independently atomic bin orders.
        z, xb, sb, snapshot = model.perceive(x, state.requires_grad_(), grid)
        results = []
        with patch.object(model, "perceive", return_value=(z, xb, sb, snapshot)):
            for compiled in (False, True):
                model.compile_step = compiled
                torch.manual_seed(13)
                output = step_euler(model, x, state, grid, intermediate=True, noise_std=0.02)
                loss = output[0].square().sum() + output[1].square().sum() + output[2]["dx_norm"]
                gradients = torch.autograd.grad(loss, (state, *model.parameters()), retain_graph=True)
                results.append((output, gradients, torch.cuda.get_rng_state()))
        for a, b in zip(results[0][0][:2], results[1][0][:2]):
            torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)
        for a, b in zip(results[0][1], results[1][1]):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(results[0][2], results[1][2], rtol=0, atol=0)
        torch._dynamo.reset()

    def test_three_dimensional_compiled_update(self):
        if self.instrumented:
            self.skipTest("Inductor runtime compilation is checked without CUDA instrumentation")
        from models.npa import NPA
        torch.manual_seed(7)
        model = NPA(spatial_dims=3, state_dims=4, hidden_dims=16).cuda()
        z = torch.randn(1, 17, model.perception_dim, device="cuda", requires_grad=True)
        x = torch.randn(1, 17, 4, device="cuda", requires_grad=True)
        s = torch.randn(1, 17, 4, device="cuda", requires_grad=True)
        mask = torch.rand(1, 17, 1, device="cuda")
        noise = torch.randn_like(x)
        inputs = (z, x, s, mask, noise, 0.1, 0.3, True, 0.3**0.5, 0.5, 0.02)
        expected = model._euler_from_perception(*inputs)
        actual = model._compiled_step(*inputs)
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, rtol=3e-5, atol=3e-6)
        variables = (z, x, s, *model.parameters())
        grads = [torch.autograd.grad(sum(value.square().sum() for value in output), variables,
                                    retain_graph=True) for output in (expected, actual)]
        for value, reference in zip(grads[1], grads[0]):
            torch.testing.assert_close(value, reference, rtol=1e-4, atol=1e-5)
        torch._dynamo.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--instrumented", action="store_true")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TexturePipelineTests)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(CPUUpdateTests))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(SplattingTests))
    if args.gpu:
        CUDAPipelineTests.instrumented = args.instrumented
        class CUDAImageTests(TexturePipelineTests):
            def test_cuda_image_adjoint(self):
                self.check_image_adjoint("cuda")
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(CUDAImageTests))
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(CUDAPipelineTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    gc.collect()
    raise SystemExit(0 if result.wasSuccessful() else 1)
