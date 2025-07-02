#!/usr/bin/env python3
"""
GPU Diagnostic Script for OpenVINO with Intel HD Graphics 520
Run this script to diagnose GPU availability and setup issues
"""

import sys
import os
import subprocess
import platform

def print_section(title):
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")

def run_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return "", str(e), -1

def check_system_info():
    print_section("SYSTEM INFORMATION")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python: {sys.version}")

def check_gpu_hardware():
    print_section("GPU HARDWARE DETECTION")
    
    if platform.system() == "Windows":
        # Windows GPU detection
        stdout, stderr, code = run_command('wmic path win32_VideoController get name')
        if code == 0:
            print("Detected GPUs:")
            for line in stdout.split('\n'):
                line = line.strip()
                if line and line != "Name":
                    print(f"  - {line}")
        
        # Check Intel driver version
        stdout, stderr, code = run_command('wmic path win32_SystemDriver where "name=\'igdkmd64\'" get version')
        if code == 0 and stdout:
            print(f"Intel Graphics Driver: {stdout.strip()}")
    
    elif platform.system() == "Linux":
        # Linux GPU detection
        stdout, stderr, code = run_command('lspci | grep -i vga')
        if code == 0:
            print("Detected GPUs:")
            for line in stdout.split('\n'):
                if line.strip():
                    print(f"  - {line}")
        
        # Check Intel GPU specifically
        stdout, stderr, code = run_command('lspci | grep -i intel')
        if code == 0:
            print("Intel devices:")
            for line in stdout.split('\n'):
                if line.strip():
                    print(f"  - {line}")

def check_opencl():
    print_section("OPENCL SUPPORT")
    
    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
        print(f"Found {len(platforms)} OpenCL platform(s):")
        
        for i, platform in enumerate(platforms):
            print(f"  Platform {i}: {platform.name}")
            devices = platform.get_devices()
            for j, device in enumerate(devices):
                print(f"    Device {j}: {device.name} ({device.type})")
                print(f"      Max compute units: {device.max_compute_units}")
                print(f"      Global memory: {device.global_mem_size // (1024*1024)} MB")
    
    except ImportError:
        print("PyOpenCL not installed. Install with: pip install pyopencl")
    except Exception as e:
        print(f"OpenCL error: {e}")

def check_onnxruntime():
    print_section("ONNX RUNTIME PROVIDERS")
    
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print("Available providers:")
        for provider in providers:
            print(f"  ✓ {provider}")
        
        if 'OpenVINOExecutionProvider' not in providers:
            print("\n❌ OpenVINOExecutionProvider NOT FOUND!")
            print("Solutions:")
            print("  1. Install: pip install onnxruntime-openvino")
            print("  2. Or use: pip install onnxruntime-gpu")
        else:
            print("\n✅ OpenVINOExecutionProvider available")
    
    except ImportError:
        print("ONNX Runtime not installed. Install with: pip install onnxruntime")

def check_openvino():
    print_section("OPENVINO INSTALLATION")
    
    try:
        import openvino as ov
        print(f"OpenVINO version: {ov.__version__}")
        
        # Check available devices
        core = ov.Core()
        devices = core.available_devices
        print("Available OpenVINO devices:")
        for device in devices:
            print(f"  - {device}")
            
        # Try to check GPU specifically
        if 'GPU' in devices:
            try:
                gpu_info = core.get_property('GPU', 'FULL_DEVICE_NAME')
                print(f"GPU Device: {gpu_info}")
            except:
                print("GPU device found but cannot get details")
        else:
            print("❌ No GPU device found in OpenVINO")
    
    except ImportError:
        print("OpenVINO not installed. Install with: pip install openvino")
    except Exception as e:
        print(f"OpenVINO error: {e}")

def check_intel_drivers():
    print_section("INTEL GRAPHICS DRIVERS")
    
    if platform.system() == "Windows":
        print("Check Intel Graphics drivers in Device Manager")
        print("Update from: https://www.intel.com/content/www/us/en/support/articles/000005629/graphics.html")
    
    elif platform.system() == "Linux":
        # Check if Intel compute runtime is installed
        stdout, stderr, code = run_command('dpkg -l | grep intel')
        if code == 0 and stdout:
            print("Intel packages found:")
            for line in stdout.split('\n'):
                if 'intel' in line.lower():
                    print(f"  {line}")
        
        # Check OpenCL ICD
        stdout, stderr, code = run_command('ls /etc/OpenCL/vendors/')
        if code == 0:
            print("OpenCL vendors:")
            for line in stdout.split('\n'):
                if line.strip():
                    print(f"  - {line}")

def test_simple_inference():
    print_section("SIMPLE INFERENCE TEST")
    
    try:
        import onnxruntime as ort
        import numpy as np
        
        # Create a simple test
        providers_to_test = [
            ['OpenVINOExecutionProvider', 'CPUExecutionProvider'],
            ['CPUExecutionProvider']
        ]
        
        for providers in providers_to_test:
            try:
                print(f"Testing with providers: {providers}")
                
                # We can't test without an actual model, but we can test session creation
                available = [p for p in providers if p in ort.get_available_providers()]
                print(f"  Available from list: {available}")
                
                if 'OpenVINOExecutionProvider' in available:
                    print("  ✅ OpenVINO provider ready for testing")
                else:
                    print("  ❌ OpenVINO provider not available")
                
            except Exception as e:
                print(f"  Error testing {providers}: {e}")
    
    except ImportError:
        print("Cannot test - ONNX Runtime not available")

def provide_solutions():
    print_section("SOLUTIONS FOR INTEL HD GRAPHICS 520")
    
    print("🔧 SETUP STEPS:")
    print("\n1. Install ONNX Runtime with OpenVINO:")
    print("   pip uninstall onnxruntime")
    print("   pip install onnxruntime-openvino")
    
    print("\n2. Install Intel OpenVINO Runtime:")
    print("   pip install openvino")
    
    if platform.system() == "Windows":
        print("\n3. Update Intel Graphics Driver:")
        print("   - Download from Intel website")
        print("   - Version 30.0.100.9955 or later required")
        print("   - Reboot after installation")
    
    elif platform.system() == "Linux":
        print("\n3. Install Intel Compute Runtime:")
        print("   sudo apt update")
        print("   sudo apt install intel-opencl-icd intel-level-zero-gpu level-zero")
        print("   sudo apt install ocl-icd-libopencl1")
    
    print("\n4. Environment Variables:")
    print("   export OPENVINO_VERBOSE=1")
    print("   export OV_GPU_CACHE_MODEL=1")
    
    print("\n⚠️  IMPORTANT LIMITATIONS:")
    print("- Intel HD Graphics 520 (Skylake GT2) has LIMITED OpenVINO support")
    print("- GPU acceleration may not work optimally with this older hardware")
    print("- Consider CPU execution as primary option")
    print("- Modern Intel GPUs (Gen9+) have better OpenVINO support")

def main():
    print("🔍 GPU DIAGNOSTIC FOR OPENVINO + INTEL HD GRAPHICS 520")
    print("=" * 60)
    
    check_system_info()
    check_gpu_hardware()
    check_opencl()
    check_onnxruntime()
    check_openvino()
    check_intel_drivers()
    test_simple_inference()
    provide_solutions()
    
    print("\n" + "="*60)
    print("✅ DIAGNOSTIC COMPLETE")
    print("📧 Share this output if you need further help")

if __name__ == "__main__":
    main()