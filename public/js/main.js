class MNISTApp {
  // Initialise l'application MNIST
  constructor() {
    this.canvas = document.getElementById('canvas');
    this.canvas.width = 280;
    this.canvas.height = 280;
    this.ctx = this.canvas.getContext('2d');
    this.ctx.fillStyle = 'black';
    this.ctx.fillRect(0, 0, 280, 280);
    this.isDrawing = false;
    this.lastX = 0;
    this.lastY = 0;
    this.model = null;
    
    this.setupCanvas();
    this.setupButtons();
    this.loadModel();
  }

  // Configure les événements de dessin sur le canvas (souris et tactile)
  setupCanvas() {
    // Mouse events for desktop
    this.canvas.addEventListener('mousedown', (e) => {
      this.isDrawing = true;
      const rect = this.canvas.getBoundingClientRect();
      this.lastX = e.clientX - rect.left;
      this.lastY = e.clientY - rect.top;
    });

    this.canvas.addEventListener('mousemove', (e) => {
      if (this.isDrawing) {
        this.draw(e);
      }
    });

    this.canvas.addEventListener('mouseup', () => {
      this.isDrawing = false;
    });

    this.canvas.addEventListener('mouseout', () => {
      this.isDrawing = false;
    });
  }

  // Dessine une ligne continue sur le canvas
  draw(e) {
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    this.ctx.globalCompositeOperation = 'source-over';
    this.ctx.strokeStyle = 'white';
    this.ctx.lineWidth = 20;
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';
    
    this.ctx.beginPath();
    this.ctx.moveTo(this.lastX, this.lastY);
    this.ctx.lineTo(x, y);
    this.ctx.stroke();
    
    this.lastX = x;
    this.lastY = y;
  }

  // Configure les boutons "Effacer" et "Prédire"
  setupButtons() {
    document.getElementById('clear').addEventListener('click', () => {
      this.ctx.fillStyle = 'black';
      this.ctx.fillRect(0, 0, 280, 280);
      document.getElementById('result').textContent = '';
      document.getElementById('bars').innerHTML = '';
    });

    document.getElementById('predict').addEventListener('click', () => {
      this.predict();
    });
  }

  // Charge le modèle ONNX depuis le fichier
  async loadModel() {
    try {
      this.model = await ort.InferenceSession.create('/models/mnist.onnx');
      console.log('Model loaded successfully');
    } catch (error) {
      console.error('Error loading model:', error);
      alert('Erreur: Impossible de charger le modèle. Vérifiez que le fichier mnist.onnx est présent dans le dossier /public/models/');
    }
  }

  // Prépare l'image du canvas pour la prédiction (redimensionne et normalise)
  preprocessCanvas() {
    const imageData = this.ctx.getImageData(0, 0, 280, 280);
    const resized = this.resizeImageData(imageData, 28, 28);
    
    const input = new Float32Array(28 * 28);
    for (let i = 0; i < 28 * 28; i++) {
      const pixelIndex = i * 4;
      const gray = resized.data[pixelIndex] / 255.0;
      input[i] = (gray - 0.1307) / 0.3081;
    }
    
    return input;
  }

  // Redimensionne les données d'image à la taille spécifiée
  resizeImageData(imageData, width, height) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = width;
    canvas.height = height;
    
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    tempCanvas.width = imageData.width;
    tempCanvas.height = imageData.height;
    tempCtx.putImageData(imageData, 0, 0);
    
    ctx.drawImage(tempCanvas, 0, 0, width, height);
    return ctx.getImageData(0, 0, width, height);
  }

  // Effectue une prédiction sur le dessin et affiche le résultat
  async predict() {
    if (!this.model) return;

    try {
      const inputData = this.preprocessCanvas();
      const inputTensor = new ort.Tensor('float32', inputData, [1, 1, 28, 28]);
      const feeds = { input: inputTensor };
      const results = await this.model.run(feeds);
      const output = results.output.data;

      const maxIndex = output.indexOf(Math.max(...output));
      document.getElementById('result').textContent = `Predicted: ${maxIndex}`;

      this.displayConfidence(output);
    } catch (error) {
      console.error('Prediction error:', error);
    }
  }

  // Affiche les barres de confiance pour chaque chiffre (0-9)
  displayConfidence(output) {
    const softmax = this.softmax(Array.from(output));
    const barsContainer = document.getElementById('bars');
    barsContainer.innerHTML = '';

    softmax.forEach((confidence, digit) => {
      const bar = document.createElement('div');
      bar.className = 'flex items-center gap-2';
      bar.innerHTML = `
        <span class="w-4 text-xs">${digit}</span>
        <div class="flex-1 bg-gray-200 rounded h-3">
          <div class="bg-blue-500 h-3 rounded" style="width: ${confidence * 100}%"></div>
        </div>
        <span class="text-xs w-12">${(confidence * 100).toFixed(1)}%</span>
      `;
      barsContainer.appendChild(bar);
    });
  }

  // Calcule la fonction softmax pour convertir les scores en probabilités
  softmax(arr) {
    const max = Math.max(...arr);
    const exp = arr.map(x => Math.exp(x - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map(x => x / sum);
  }
}

new MNISTApp();
