import os
import json
import numpy as np
import tensorflow as tf

def export_model_weights():
    print("Loading digit_model.keras...")
    model = tf.keras.models.load_model("digit_model.keras")

    # Extract weights for Dense layers
    layers_weights = []
    for layer in model.layers:
        weights = layer.get_weights()
        if weights: # Layer has weights and biases
            w, b = weights
            layers_weights.append({
                "name": layer.name,
                "w": w.tolist(),
                "b": b.tolist()
            })

    os.makedirs("public", exist_ok=True)
    out_path = os.path.join("public", "weights.json")
    with open(out_path, "w") as f:
        json.dump(layers_weights, f)

    print(f"Successfully exported model weights to {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)")

if __name__ == "__main__":
    export_model_weights()
