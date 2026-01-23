#!/usr/bin/env python3
"""Test script to verify dataloaders on Modal.

This script tests create_dataloaders as used in train.py and validates
that the data shapes and types are correct for model training.
"""

import sys
from pathlib import Path

import modal

APP_NAME = "test-dataloader"
WORKDIR = "/workspace"
DATASET_MOUNT = "/data/signaltrain"

# Persistent volume for the SignalTrain dataset
signaltrain_volume = modal.Volume.from_name(
    "signaltrain-dataset", create_if_missing=False
)

# Base image with dependencies matching modal_app.py
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        # Core deps
        "numpy>=2.3.3",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "h5py>=3.8.0",
        "scipy>=1.11.0",
        "pyyaml>=6.0.0",
        # Torch stack
        "torch>=2.1.2",
        "torchvision>=0.16.1",
    )
    # Include project code so imports resolve
    .add_local_dir(Path(__file__).parent, remote_path=WORKDIR)
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    volumes={DATASET_MOUNT: signaltrain_volume},
    timeout=60 * 10,  # 10 minutes
)
def test_dataloaders(
    root_dir: str = "/data/signaltrain/SignalTrain_LA2A_Dataset_1.1",
    train_subset: str = "Train",
    val_subset: str = "Val",
    test_subset: str = "Test",
    train_length: int = 65536,
    eval_length: int = 131072,
    batch_size: int = 8,
    num_workers: int = 4,
    n_params: int = 2,
    num_batches_to_test: int = 5,
):
    """Test dataloaders and validate data shapes/types as used in train.py."""
    import sys
    import torch
    
    sys.path.insert(0, WORKDIR)

    from dataloader import DatasetConfig as DataConfigForLoader, create_dataloaders

    print("=" * 80)
    print("DATALOADER TEST ON MODAL")
    print("=" * 80)
    print(f"Dataset root: {root_dir}")
    print(f"Train subset: {train_subset}")
    print(f"Val subset: {val_subset}")
    print(f"Test subset: {test_subset}")
    print(f"Train length: {train_length}")
    print(f"Eval length: {eval_length}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Num params: {n_params}")
    print(f"Testing {num_batches_to_test} batches from each loader")
    print("=" * 80)

    # Create data configuration exactly as in train.py
    data_config = DataConfigForLoader(
        root_dir=root_dir,
        train_subset=train_subset,
        val_subset=val_subset,
        test_subset=test_subset,
        train_length=train_length,
        eval_length=eval_length,
        batch_size=batch_size,
        num_workers=num_workers,
        n_params=n_params,
        preload=False,
        half_precision=False,
        shuffle=True,
        pin_memory=True,
    )

    print("\nCreating dataloaders...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(data_config)
        print("✓ Dataloaders created successfully")
    except Exception as e:
        print(f"❌ Failed to create dataloaders: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

    # Test train loader
    print("\n" + "=" * 80)
    print("TESTING TRAIN LOADER")
    print("=" * 80)
    train_results = test_loader_batches(
        train_loader, "train", num_batches_to_test, train_length, n_params
    )

    # Test val loader
    print("\n" + "=" * 80)
    print("TESTING VALIDATION LOADER")
    print("=" * 80)
    val_results = test_loader_batches(
        val_loader, "val", num_batches_to_test, eval_length, n_params
    )

    # Test test loader if available
    test_results = None
    if test_loader is not None:
        print("\n" + "=" * 80)
        print("TESTING TEST LOADER")
        print("=" * 80)
        test_results = test_loader_batches(
            test_loader, "test", num_batches_to_test, eval_length, n_params
        )
    else:
        print("\n⚠ Test loader not available (test subset may not exist)")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    all_success = train_results["success"] and val_results["success"]
    if test_results:
        all_success = all_success and test_results["success"]

    print(f"Train loader: {'✓ PASS' if train_results['success'] else '❌ FAIL'}")
    print(f"Val loader: {'✓ PASS' if val_results['success'] else '❌ FAIL'}")
    if test_results:
        print(f"Test loader: {'✓ PASS' if test_results['success'] else '❌ FAIL'}")
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_success else '❌ SOME TESTS FAILED'}")

    return {
        "success": all_success,
        "train": train_results,
        "val": val_results,
        "test": test_results,
    }


def test_loader_batches(loader, loader_name, num_batches, expected_length, n_params):
    """Test batches from a dataloader and validate shapes/types."""
    import torch
    import numpy as np

    results = {
        "success": True,
        "batches_tested": 0,
        "errors": [],
    }

    try:
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= num_batches:
                break

            print(f"\n--- Batch {batch_idx + 1} ---")
            input_audio, target_audio, params = batch

            # Validate shapes
            print(f"Input audio shape: {input_audio.shape}")
            print(f"Target audio shape: {target_audio.shape}")
            print(f"Params shape: {params.shape}")

            # Expected shapes:
            # - input_audio: (B, 1, T)
            # - target_audio: (B, 1, T)
            # - params: (B, P)
            B = input_audio.shape[0]
            T = input_audio.shape[2]

            # Validate input_audio
            if input_audio.dim() != 3:
                error = f"Input audio should be 3D (B, 1, T), got {input_audio.dim()}D"
                print(f"❌ {error}")
                results["errors"].append(error)
                results["success"] = False
            elif input_audio.shape[1] != 1:
                error = f"Input audio should have 1 channel, got {input_audio.shape[1]}"
                print(f"❌ {error}")
                results["errors"].append(error)
                results["success"] = False
            elif T != expected_length:
                error = f"Input audio length should be {expected_length}, got {T}"
                print(f"❌ {error}")
                results["errors"].append(error)
                results["success"] = False
            else:
                print(f"✓ Input audio shape valid: (B={B}, C=1, T={T})")

            # Validate target_audio
            if target_audio.shape != input_audio.shape:
                error = f"Target audio shape {target_audio.shape} != input shape {input_audio.shape}"
                print(f"❌ {error}")
                results["errors"].append(error)
                results["success"] = False
            else:
                print(f"✓ Target audio shape matches input")

            # Validate params
            if params.dim() != 2:
                error = f"Params should be 2D (B, P), got {params.dim()}D"
                print(f"❌ {error}")
                results["errors"].append(error)
                results["success"] = False
            elif params.shape[0] != B:
                error = f"Params batch size {params.shape[0]} != audio batch size {B}"
                print(f"❌ {error}")
                results["errors"].append(error)
                results["success"] = False
            elif params.shape[1] != n_params:
                error = f"Params should have {n_params} params, got {params.shape[1]}"
                print(f"❌ {error}")
                results["errors"].append(error)
                results["success"] = False
            else:
                print(f"✓ Params shape valid: (B={B}, P={n_params})")

            # Validate data types
            print(f"\nData types:")
            print(f"  Input audio dtype: {input_audio.dtype}")
            print(f"  Target audio dtype: {target_audio.dtype}")
            print(f"  Params dtype: {params.dtype}")

            if input_audio.dtype != torch.float32:
                error = f"Input audio should be float32, got {input_audio.dtype}"
                print(f"❌ {error}")
                results["errors"].append(error)
                results["success"] = False
            else:
                print(f"✓ Input audio dtype is float32")

            if target_audio.dtype != torch.float32:
                error = f"Target audio should be float32, got {target_audio.dtype}"
                print(f"❌ {error}")
                results["errors"].append(error)
                results["success"] = False
            else:
                print(f"✓ Target audio dtype is float32")

            if params.dtype != torch.float32:
                error = f"Params should be float32, got {params.dtype}"
                print(f"❌ {error}")
                results["errors"].append(error)
                results["success"] = False
            else:
                print(f"✓ Params dtype is float32")

            # Validate value ranges
            print(f"\nValue ranges:")
            input_min, input_max = input_audio.min().item(), input_audio.max().item()
            target_min, target_max = target_audio.min().item(), target_audio.max().item()
            params_min, params_max = params.min().item(), params.max().item()

            print(f"  Input audio: [{input_min:.6f}, {input_max:.6f}]")
            print(f"  Target audio: [{target_min:.6f}, {target_max:.6f}]")
            print(f"  Params: [{params_min:.6f}, {params_max:.6f}]")

            # Audio should typically be in reasonable range (not necessarily [-1, 1] but not extreme)
            if abs(input_max) > 10.0 or abs(input_min) > 10.0:
                print(f"⚠ Warning: Input audio values seem extreme (outside [-10, 10])")
            else:
                print(f"✓ Input audio values in reasonable range")

            if abs(target_max) > 10.0 or abs(target_min) > 10.0:
                print(f"⚠ Warning: Target audio values seem extreme (outside [-10, 10])")
            else:
                print(f"✓ Target audio values in reasonable range")

            # Params should be in reasonable range (depends on dataset, but typically [0, 1] or similar)
            print(f"✓ Params values: [{params_min:.6f}, {params_max:.6f}]")

            # Check for NaN or Inf
            print(f"\nChecking for NaN/Inf:")
            input_has_nan = torch.isnan(input_audio).any().item()
            input_has_inf = torch.isinf(input_audio).any().item()
            target_has_nan = torch.isnan(target_audio).any().item()
            target_has_inf = torch.isinf(target_audio).any().item()
            params_has_nan = torch.isnan(params).any().item()
            params_has_inf = torch.isinf(params).any().item()

            if input_has_nan or input_has_inf:
                error = f"Input audio contains {'NaN' if input_has_nan else 'Inf'}"
                print(f"❌ {error}")
                results["errors"].append(error)
                results["success"] = False
            else:
                print(f"✓ Input audio: no NaN/Inf")

            if target_has_nan or target_has_inf:
                error = f"Target audio contains {'NaN' if target_has_nan else 'Inf'}"
                print(f"❌ {error}")
                results["errors"].append(error)
                results["success"] = False
            else:
                print(f"✓ Target audio: no NaN/Inf")

            if params_has_nan or params_has_inf:
                error = f"Params contain {'NaN' if params_has_nan else 'Inf'}"
                print(f"❌ {error}")
                results["errors"].append(error)
                results["success"] = False
            else:
                print(f"✓ Params: no NaN/Inf")

            # Check contiguity (important for LSTM)
            print(f"\nMemory layout:")
            print(f"  Input audio contiguous: {input_audio.is_contiguous()}")
            print(f"  Target audio contiguous: {target_audio.is_contiguous()}")
            print(f"  Params contiguous: {params.is_contiguous()}")

            # Test that data can be used in a forward pass (simulate model input)
            print(f"\nTesting model input compatibility:")
            try:
                # Simulate what the model expects: (B, 1, T) for audio, (B, P) for params
                # Check that we can transpose as the model does
                seq = input_audio.transpose(1, 2).contiguous()  # (B, T, 1)
                if params is not None and n_params > 0:
                    p = params.unsqueeze(1).repeat(1, seq.shape[1], 1)  # (B, T, P)
                    combined = torch.cat([seq, p], dim=-1)  # (B, T, 1+P)
                    combined = combined.contiguous()
                    print(f"✓ Combined input shape: {combined.shape}")
                print(f"✓ Data is compatible with model forward pass")
            except Exception as e:
                error = f"Failed to prepare data for model: {e}"
                print(f"❌ {error}")
                results["errors"].append(error)
                results["success"] = False

            results["batches_tested"] += 1

        print(f"\n✓ Tested {results['batches_tested']} batches from {loader_name} loader")

    except Exception as e:
        error = f"Error testing {loader_name} loader: {e}"
        print(f"❌ {error}")
        import traceback
        traceback.print_exc()
        results["errors"].append(error)
        results["success"] = False

    return results


@app.local_entrypoint()
def main(
    root_dir: str = "/data/signaltrain/SignalTrain_LA2A_Dataset_1.1",
    train_subset: str = "Train",
    val_subset: str = "Val",
    test_subset: str = "Test",
    train_length: int = 65536,
    eval_length: int = 131072,
    batch_size: int = 8,
    num_workers: int = 4,
    n_params: int = 2,
    num_batches: int = 5,
):
    """Local entrypoint to run dataloader tests on Modal."""
    result = test_dataloaders.remote(
        root_dir=root_dir,
        train_subset=train_subset,
        val_subset=val_subset,
        test_subset=test_subset,
        train_length=train_length,
        eval_length=eval_length,
        batch_size=batch_size,
        num_workers=num_workers,
        n_params=n_params,
        num_batches_to_test=num_batches,
    )

    print("\n" + "=" * 80)
    print("FINAL RESULT")
    print("=" * 80)
    import json
    print(json.dumps(result, indent=2, default=str))

    if result.get("success"):
        print("\n✓ All dataloader tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some dataloader tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
