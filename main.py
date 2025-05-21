import argparse
import os
import torch

def run_train():
    print("🚀 Verificando entorno CUDA y GPU...")
    print(f"🧠 CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🖥️ GPU en uso: {torch.cuda.get_device_name(0)}")

        # Tensor de prueba
        x = torch.rand(3, 3).to("cuda")
        print("✅ Tensor de prueba en CUDA:")
        print(x)
        print(f"📐 Forma: {x.shape} | Tipo: {x.dtype} | Dispositivo: {x.device}")
    else:
        print("⚠️ CUDA NO disponible, usando CPU.")

    print("\n▶️ Iniciando entrenamiento...\n")
    os.system("python train_transformer.py")
    print("\n▶️ Iniciando entrenamiento...\n")

def run_predict():
    os.system("python predict.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconocimiento de Señas en LSC")
    parser.add_argument("--modo", type=str, choices=["train", "predict"], required=True,
                        help="Modo de ejecución: 'train' o 'predict'")
    args = parser.parse_args()

    if args.modo == "train":
        run_train()
    elif args.modo == "predict":
        run_predict()
