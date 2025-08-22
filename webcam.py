import argparse
from collections import deque
import time
import cv2
import numpy as np

from model import Model

# Ajuste dos rótulos se não usar settings.yaml binário
VIOLENCE_LABELS = {
    "violence", "fight", "fighting", "assault", "aggression", "weapon", "knife", "gun"
}

def is_violence_from_label(label: str, binary_mode: bool) -> bool:
    if label is None:
        return False
    L = label.strip().lower()
    if binary_mode:
        return L.startswith("violence")
    return any(k in L for k in VIOLENCE_LABELS)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="Índice da câmera (C922 geralmente é 0 se for única)")
    ap.add_argument("--binary-settings", action="store_true",
                    help="Use se você deixou o settings.yaml com só 2 rótulos (Violence / Non-violence)")
    ap.add_argument("--record", type=str, default="", help="Caminho para salvar vídeo (opcional, ex.: out.mp4)")
    args = ap.parse_args()

    # Configurações otimizadas para Logitech C922 Pro
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    FRAME_FPS = 15

    print(f"Iniciando Logitech C922 Pro em {FRAME_WIDTH}x{FRAME_HEIGHT} @ {FRAME_FPS}fps")

    model = Model()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Não consegui abrir a câmera {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)

    writer = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.record, fourcc, FRAME_FPS, (FRAME_WIDTH, FRAME_HEIGHT))

    # janela de suavização
    hist = deque(maxlen=12)
    last_time = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.time()
            if (now - last_time) < (1.0 / FRAME_FPS):
                continue
            last_time = now

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out = model.predict(image=rgb)
            label = (out or {}).get("label", None)

            violence = is_violence_from_label(label, args.binary_settings)
            hist.append(1 if violence else 0)
            votes = sum(hist)
            triggered = votes >= 8  # mínimo de 8 votos de violência em 12 frames

            display = frame.copy()
            h, w = display.shape[:2]
            verdict = "VIOLENCE" if violence else "NON-VIOLENCE"
            color = (0, 0, 255) if triggered else ((0, 0, 255) if violence else (0, 200, 0))

            cv2.rectangle(display, (10, 10), (w - 10, 90), (0, 0, 0), -1)
            cv2.putText(display, f"Pred: {label or 'n/a'}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display, f"Binary: {verdict}  (votes {votes}/{len(hist)})",
                        (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

            if triggered:
                cv2.rectangle(display, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)

            if writer is not None:
                writer.write(display)

            cv2.imshow("Violence Detection - Logitech C922 Pro", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
