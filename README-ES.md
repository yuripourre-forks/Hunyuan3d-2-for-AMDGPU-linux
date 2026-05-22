# Hunyuan3D-2 para AMDGPU en Linux

Scripts para instalar y ejecutar [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) en Linux con GPU AMD y ROCm. La generación de texturas requiere compilar `custom_rasterizer` para HIP; este repositorio lo construye desde el código fuente durante la instalación.

## Requisitos

1. **ROCm** — [Instalar ROCm en Linux](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html)
2. **Paquetes del sistema** — `git`, `python3`, `patch`, compilador C++, `ninja-build`, `rocm-dev`
3. **Python** — 3.10 o superior
4. **Ruta sin espacios** — No instalar en rutas con espacios ([ROCm #4329](https://github.com/ROCm/ROCm/issues/4329))

## Instalación

```bash
git clone https://github.com/dgarcia1985/Hunyuan3d-2-for-AMDGPU-linux.git
cd Hunyuan3d-2-for-AMDGPU-linux
./scripts/install.sh
```

Opciones: `--port`, `--gpu-arch`, `--skip-flash-attention`

## Ejecución

```bash
./scripts/run.sh
./scripts/run-multiview.sh
```

## Compilar custom_rasterizer manualmente

```bash
source .venv/bin/activate
export GPU_ARCHS=gfx1100
./scripts/build-custom-rasterizer.sh
```

Ver [README.md](README.md) (inglés) para detalles de configuración y problemas conocidos.
