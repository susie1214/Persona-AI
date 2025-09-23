#!/usr/bin/env python3
"""
run_tests.py
Persona-AI Meeting Assistant 통합 테스트 실행기

각 모듈을 독립적으로 테스트하고 전체 시스템의 상태를 확인합니다.
"""

import sys
import os
import subprocess
import time
from pathlib import Path

def print_header(title):
    """테스트 헤더 출력"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

def print_section(title):
    """섹션 헤더 출력"""
    print(f"\n{title}")
    print("-" * len(title))

def run_module_test(module_name, description):
    """개별 모듈 테스트 실행"""
    print(f"\n🔍 Testing {module_name} - {description}")
    print("   " + "-" * 50)
    
    try:
        # Python 모듈로 직접 실행
        result = subprocess.run([
            sys.executable, f"{module_name}.py"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("   ✅ PASSED")
            if result.stdout:
                # 출력의 마지막 몇 줄만 표시
                lines = result.stdout.strip().split('\n')
                if len(lines) > 10:
                    print("   Output (last 10 lines):")
                    for line in lines[-10:]:
                        print(f"     {line}")
                else:
                    print("   Output:")
                    for line in lines:
                        print(f"     {line}")
            return True
        else:
            print("   ❌ FAILED")
            if result.stderr:
                print("   Error:")
                for line in result.stderr.strip().split('\n')[:5]:
                    print(f"     {line}")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ⏰ TIMEOUT - Test took too long")
        return False
    except FileNotFoundError:
        print(f"   📁 FILE NOT FOUND - {module_name}.py")
        return False
    except Exception as e:
        print(f"   💥 ERROR - {e}")
        return False

def check_dependencies():
    """의존성 패키지 확인"""
    print_section("📦 Dependency Check")
    
    required_packages = [
        ("PyQt6", "GUI framework"),
        ("numpy", "Numerical computing"),
        ("soundfile", "Audio file I/O"),
        ("pyaudio", "Audio input/output"),
        ("faster_whisper", "Speech recognition"),
        ("pyannote.audio", "Speaker diarization"),
        ("torch", "Deep learning framework"),
        ("qdrant_client", "Vector database"),
        ("sentence_transformers", "Text embeddings"),
        ("dateparser", "Date parsing"),
        ("sklearn", "Machine learning utilities"),
    ]
    
    missing_packages = []
    
    for package, description in required_packages:
        try:
            if package == "sklearn":
                import sklearn
            elif package == "pyannote.audio":
                from pyannote.audio import Pipeline
            else:
                __import__(package)
            print(f"   ✅ {package:<25} - {description}")
        except ImportError:
            print(f"   ❌ {package:<25} - {description} (MISSING)")
            missing_packages.append(package)
    
    optional_packages = [
        ("psutil", "System monitoring"),
        ("pandas", "Data analysis"),
        ("matplotlib", "Plotting utilities"),
    ]
    
    print("\n   Optional packages:")
    for package, description in optional_packages:
        try:
            __import__(package)
            print(f"   ✅ {package:<25} - {description}")
        except ImportError:
            print(f"   ⚠️  {package:<25} - {description} (optional)")
    
    return missing_packages

def check_system_resources():
    """시스템 리소스 확인"""
    print_section("💻 System Resources Check")
    
    try:
        import psutil
        
        # CPU 정보
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        print(f"   CPU: {cpu_count} cores")
        if cpu_freq:
            print(f"   CPU Frequency: {cpu_freq.current:.0f} MHz")
        
        # 메모리 정보
        memory = psutil.virtual_memory()
        print(f"   Memory: {memory.total // (1024**3)} GB total, {memory.available // (1024**3)} GB available")
        
        # 디스크 정보
        disk = psutil.disk_usage('.')
        print(f"   Disk: {disk.free // (1024**3)} GB free of {disk.total // (1024**3)} GB")
        
        # GPU 확인 (간단히)
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                print(f"   GPU: {gpu_count}x {gpu_name}")
            else:
                print("   GPU: Not available or not CUDA-enabled")
        except ImportError:
            print("   GPU: Cannot check (torch not available)")
        
    except ImportError:
        print("   ⚠️ psutil not available - limited system info")
        print(f"   Python: {sys.version}")
        print(f"   Platform: {sys.platform}")

def main():
    """메인 테스트 실행"""
    print_header("Persona-AI Meeting Assistant - Comprehensive Test Suite")
    
    start_time = time.time()
    
    # 작업 디렉토리 확인
    current_dir = Path.cwd()
    print(f"Working directory: {current_dir}")
    
    # 시스템 리소스 확인
    check_system_resources()
    
    # 의존성 확인
    missing_deps = check_dependencies()
    
    if missing_deps:
        print_section("⚠️ Missing Dependencies")
        print("   The following required packages are missing:")
        for dep in missing_deps:
            print(f"     - {dep}")
        print("\n   Install with: pip install -r requirements.txt")
    
    # 모듈 테스트 정의
    modules_to_test = [
        ("config", "Configuration Management"),
        ("models", "Data Models & Structures"),
        ("audio_processor", "Audio Processing & STT"),
        ("rag_manager", "RAG System & Embeddings"),
        ("meeting_analyzer", "Meeting Analysis & Summarization"),
        ("main_application", "Main Application (with --test flag)"),
    ]
    
    print_section("🧪 Individual Module Tests")
    
    test_results = {}
    
    for module_name, description in modules_to_test:
        if module_name == "main_application":
            # 메인 애플리케이션은 특별한 플래그로 테스트
            print(f"\n🔍 Testing {module_name} - {description}")
            print("   " + "-" * 50)
            try:
                result = subprocess.run([
                    sys.executable, f"{module_name}.py", "--test"
                ], capture_output=True, text=True, timeout=60)
                
                success = result.returncode == 0
                test_results[module_name] = success
                
                if success:
                    print("   ✅ PASSED")
                else:
                    print("   ❌ FAILED")
                    if result.stderr:
                        print("   Error:")
                        for line in result.stderr.strip().split('\n')[:3]:
                            print(f"     {line}")
                
            except Exception as e:
                print(f"   💥 ERROR - {e}")
                test_results[module_name] = False
        else:
            test_results[module_name] = run_module_test(module_name, description)
    
    # 통합 테스트 (간단한 버전)
    print_section("🔗 Integration Tests")
    
    integration_passed = 0
    integration_total = 0
    
    # 설정 -> 모델 연동 테스트
    integration_total += 1
    try:
        from config import config
        from models import MeetingState
        
        meeting = MeetingState()
        meeting.use_gpu = config.model.WHISPER_DEVICE == "cuda"
        print("   ✅ Config -> Models integration")
        integration_passed += 1
    except Exception as e:
        print(f"   ❌ Config -> Models integration failed: {e}")
    
    # 모델 -> RAG 연동 테스트
    integration_total += 1
    try:
        from models import Segment, ConversationEntry
        from rag_manager import ConversationManager
        
        # Mock RAG manager
        class MockRAG:
            def add_conversation_entry(self, entry): return True
            def search(self, query, limit=5, session_id=None): return []
        
        conv_mgr = ConversationManager(MockRAG())
        test_seg = Segment(0, 5, "테스트", "S1", "테스터")
        entry = conv_mgr.add_segment(test_seg)
        
        print("   ✅ Models -> RAG integration")
        integration_passed += 1
    except Exception as e:
        print(f"   ❌ Models -> RAG integration failed: {e}")
    
    # 분석기 연동 테스트
    integration_total += 1
    try:
        from models import Segment
        from meeting_analyzer import MeetingAnalyzer
        
        analyzer = MeetingAnalyzer()
        test_segments = [Segment(0, 5, "프로젝트를 완료해야 합니다", "S1", "테스터")]
        result = analyzer.analyze_meeting(test_segments)
        
        if 'action_items' in result and 'summary' in result:
            print("   ✅ Models -> Analyzer integration")
            integration_passed += 1
        else:
            print("   ❌ Models -> Analyzer integration: incomplete result")
            
    except Exception as e:
        print(f"   ❌ Models -> Analyzer integration failed: {e}")
    
    # 결과 요약
    print_section("📊 Test Results Summary")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    module_passed = sum(1 for result in test_results.values() if result)
    module_total = len(test_results)
    
    print(f"   Module Tests: {module_passed}/{module_total} passed")
    print(f"   Integration Tests: {integration_passed}/{integration_total} passed")
    print(f"   Total Time: {total_time:.2f} seconds")
    
    # 개별 모듈 결과
    print("\n   Module Results:")
    for module, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"     {module:<20} {status}")
    
    # 전체 성공 여부
    overall_success = (
        module_passed == module_total and 
        integration_passed == integration_total and
        len(missing_deps) == 0
    )
    
    if overall_success:
        print("\n   🎉 ALL TESTS PASSED - System is ready!")
        return_code = 0
    else:
        print("\n   ⚠️ SOME TESTS FAILED - Check individual results")
        return_code = 1
        
        if missing_deps:
            print(f"   📦 Missing dependencies: {len(missing_deps)}")
        if module_passed < module_total:
            print(f"   🧪 Failed modules: {module_total - module_passed}")
        if integration_passed < integration_total:
            print(f"   🔗 Failed integrations: {integration_total - integration_passed}")
    
    # 권장사항 출력
    print_section("💡 Recommendations")
    
    if missing_deps:
        print("   1. Install missing dependencies:")
        print("      pip install -r requirements.txt")
    
    if not overall_success:
        print("   2. Run individual module tests for detailed error info:")
        for module, passed in test_results.items():
            if not passed:
                print(f"      python {module}.py")
    
    if overall_success:
        print("   1. System is ready! You can run the main application:")
        print("      python main_application.py")
        print("   2. For a quick demo:")
        print("      python main_application.py --demo")
    
    print_header("Test Suite Complete")
    
    return return_code

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTest runner failed: {e}")
        sys.exit(1)