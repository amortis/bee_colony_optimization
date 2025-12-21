"""Тест GPU для проверки установки CuPy"""
import sys

print(f"Python версия: {sys.version}")
print(f"Python путь: {sys.executable}")

try:
    import cupy as cp
    print("✅ CuPy установлена успешно")
    print(f"   CuPy версия: {cp.__version__}")
    print(f"   Количество GPU: {cp.cuda.runtime.getDeviceCount()}")
    print(f"   Имя GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    
    # Тест вычислений
    x = cp.random.rand(1000, 1000)
    y = cp.sum(x)
    print(f"   Тест вычислений: {float(y):.2f} (должно быть ~500000)")
    print("✅ GPU работает корректно!")
except ImportError as e:
    print(f"❌ CuPy не установлена или не может быть импортирована")
    print(f"   Ошибка: {e}")
    print(f"   Установите: pip install cupy-cuda12x")
    print(f"   Или проверьте, что используете правильное окружение Python")
except Exception as e:
    print(f"❌ Ошибка при работе с CuPy: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

