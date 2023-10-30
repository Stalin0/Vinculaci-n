import psutil


procesos = psutil.process_iter()

for proceso in procesos:
    try:

        if proceso.pid() > 0:
            proceso.terminate()  
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass  

print("Todos los procesos en segundo plano han sido detenidos (excepto procesos del sistema).")
