// Neural Network Model Weights & UI State
let modelWeights = null;
let isDrawing = false;

const canvas = document.getElementById('draw-canvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const previewCanvas = document.getElementById('preview-canvas');
const previewCtx = previewCanvas.getContext('2d');
const brushSizeInput = document.getElementById('brush-size');
const btnClear = document.getElementById('btn-clear');

// Setup drawing canvas
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = "white";
ctx.lineCap = "round";
ctx.lineJoin = "round";

// Initialize probability bars HTML
function initProbabilityBars() {
    const container = document.getElementById('probabilities-list');
    container.innerHTML = '';
    for (let i = 0; i < 10; i++) {
        container.innerHTML += `
            <div class="d-flex align-items-center mb-1">
                <span class="fw-bold me-2" style="width: 20px;">${i}</span>
                <div class="progress flex-grow-1" style="height: 14px;">
                    <div id="bar-${i}" class="progress-bar progress-bar-custom bg-primary" role="progressbar" style="width: 0%;"></div>
                </div>
                <span id="label-${i}" class="small text-muted ms-2" style="width: 45px; text-align: right;">0.0%</span>
            </div>
        `;
    }
}

// Load model weights from weights.json
async function loadWeights() {
    try {
        const response = await fetch('weights.json');
        modelWeights = await response.json();
        console.log('Model weights loaded successfully:', modelWeights);
    } catch (err) {
        console.error('Error loading weights.json:', err);
    }
}

// Event Listeners for drawing
function getCanvasCoords(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
        x: clientX - rect.left,
        y: clientY - rect.top
    };
}

function startDrawing(e) {
    isDrawing = true;
    const coords = getCanvasCoords(e);
    ctx.beginPath();
    ctx.moveTo(coords.x, coords.y);
    e.preventDefault();
}

function draw(e) {
    if (!isDrawing) return;
    const coords = getCanvasCoords(e);
    ctx.lineWidth = brushSizeInput.value;
    ctx.lineTo(coords.x, coords.y);
    ctx.stroke();
    e.preventDefault();
    predict();
}

function stopDrawing() {
    if (isDrawing) {
        isDrawing = false;
        ctx.closePath();
        predict();
    }
}

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseleave', stopDrawing);

canvas.addEventListener('touchstart', startDrawing);
canvas.addEventListener('touchmove', draw);
canvas.addEventListener('touchend', stopDrawing);

btnClear.addEventListener('click', () => {
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    previewCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
    resetPredictionUI();
});

function resetPredictionUI() {
    document.getElementById('pred-digit').innerText = '?';
    document.getElementById('pred-conf').innerText = 'Draw on canvas';
    for (let i = 0; i < 10; i++) {
        document.getElementById(`bar-${i}`).style.width = '0%';
        document.getElementById(`label-${i}`).innerText = '0.0%';
    }
}

// Preprocessing pipeline: crop bounding box -> 20x20 -> 28x28 -> center of mass
function preprocessCanvas() {
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imgData.data;

    let minX = canvas.width, minY = canvas.height, maxX = 0, maxY = 0;
    let hasPixel = false;

    for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
            const idx = (y * canvas.width + x) * 4;
            const val = data[idx]; // Red channel (grayscale stroke)
            if (val > 25) {
                hasPixel = true;
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
    }

    if (!hasPixel) return { tensor: null, isBlank: true };

    const cropW = maxX - minX + 1;
    const cropH = maxY - minY + 1;

    // Off-screen canvas for 20x20 scaled digit
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 20;
    tempCanvas.height = 20;
    const tempCtx = tempCanvas.getContext('2d');

    // Maintain aspect ratio
    const scale = Math.min(20 / cropW, 20 / cropH);
    const targetW = Math.round(cropW * scale);
    const targetH = Math.round(cropH * scale);
    const offsetX = Math.floor((20 - targetW) / 2);
    const offsetY = Math.floor((20 - targetH) / 2);

    tempCtx.drawImage(canvas, minX, minY, cropW, cropH, offsetX, offsetY, targetW, targetH);

    // Create 28x28 canvas
    const c28 = document.createElement('canvas');
    c28.width = 28;
    c28.height = 28;
    const ctx28 = c28.getContext('2d');
    ctx28.fillStyle = "black";
    ctx28.fillRect(0, 0, 28, 28);
    ctx28.drawImage(tempCanvas, 4, 4);

    // Compute Center of Mass
    const data28 = ctx28.getImageData(0, 0, 28, 28).data;
    let sumIntensity = 0, sumX = 0, sumY = 0;
    const grid = new Float32Array(784);

    for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
            const idx = (y * 28 + x) * 4;
            const val = data28[idx];
            grid[y * 28 + x] = val / 255.0;
            sumIntensity += val;
            sumX += x * val;
            sumY += y * val;
        }
    }

    // Shift center of mass if intensity exists
    if (sumIntensity > 0) {
        const cx = sumX / sumIntensity;
        const cy = sumY / sumIntensity;
        const shiftX = Math.round(14 - cx);
        const shiftY = Math.round(14 - cy);

        if (Math.abs(shiftX) > 0 || Math.abs(shiftY) > 0) {
            const shiftedCanvas = document.createElement('canvas');
            shiftedCanvas.width = 28;
            shiftedCanvas.height = 28;
            const sCtx = shiftedCanvas.getContext('2d');
            sCtx.fillStyle = "black";
            sCtx.fillRect(0, 0, 28, 28);
            sCtx.drawImage(c28, shiftX, shiftY);
            
            const shiftedData = sCtx.getImageData(0, 0, 28, 28).data;
            for (let i = 0; i < 784; i++) {
                grid[i] = shiftedData[i * 4] / 255.0;
            }
            // Update preview
            previewCtx.imageSmoothingEnabled = false;
            previewCtx.drawImage(shiftedCanvas, 0, 0, previewCanvas.width, previewCanvas.height);
            return { tensor: grid, isBlank: false };
        }
    }

    previewCtx.imageSmoothingEnabled = false;
    previewCtx.drawImage(c28, 0, 0, previewCanvas.width, previewCanvas.height);
    return { tensor: grid, isBlank: false };
}

// Forward Pass Neural Network Inference
function forwardPass(x) {
    if (!modelWeights) return null;

    let input = x;

    for (let i = 0; i < modelWeights.length; i++) {
        const w = modelWeights[i].w; // Matrix (in_dim, out_dim)
        const b = modelWeights[i].b; // Vector (out_dim)
        const inDim = w.length;
        const outDim = b.length;

        const output = new Float32Array(outDim);

        for (let j = 0; j < outDim; j++) {
            let sum = b[j];
            for (let k = 0; k < inDim; k++) {
                sum += input[k] * w[k][j];
            }
            // Apply activation function
            if (i < modelWeights.length - 1) {
                // ReLU
                output[j] = Math.max(0, sum);
            } else {
                // Output layer logits before Softmax
                output[j] = sum;
            }
        }
        input = output;
    }

    // Softmax activation for final layer
    let maxLogit = -Infinity;
    for (let j = 0; j < input.length; j++) {
        if (input[j] > maxLogit) maxLogit = input[j];
    }

    let expSum = 0;
    const probs = new Float32Array(input.length);
    for (let j = 0; j < input.length; j++) {
        probs[j] = Math.exp(input[j] - maxLogit);
        expSum += probs[j];
    }
    for (let j = 0; j < input.length; j++) {
        probs[j] /= expSum;
    }

    return probs;
}

// Live prediction handler
function predict() {
    const { tensor, isBlank } = preprocessCanvas();
    if (isBlank || !tensor) {
        resetPredictionUI();
        return;
    }

    const probs = forwardPass(tensor);
    if (!probs) return;

    let predDigit = 0;
    let maxProb = 0;

    for (let i = 0; i < 10; i++) {
        const prob = probs[i];
        if (prob > maxProb) {
            maxProb = prob;
            predDigit = i;
        }
        const pct = (prob * 100).toFixed(1);
        const bar = document.getElementById(`bar-${i}`);
        bar.style.width = `${pct}%`;
        document.getElementById(`label-${i}`).innerText = `${pct}%`;
        
        if (i === predDigit) {
            bar.className = 'progress-bar progress-bar-custom bg-primary';
        } else {
            bar.className = 'progress-bar progress-bar-custom bg-secondary opacity-50';
        }
    }

    document.getElementById('pred-digit').innerText = predDigit;
    document.getElementById('pred-conf').innerText = `${(maxProb * 100).toFixed(1)}% Confidence`;
}

// Initialize on page load
initProbabilityBars();
loadWeights();
