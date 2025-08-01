# 🔢 MNIST Web - Reconnaissance de Chiffres Manuscrits

Une application web interactive utilisant l'intelligence artificielle pour reconnaître les chiffres manuscrits en temps réel.

![Demo](https://img.shields.io/badge/Demo-Live-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-orange)

## ✨ Fonctionnalités

- 🎨 **Interface de dessin interactive** - Dessinez directement sur le canvas
- 🧠 **Prédiction IA en temps réel** - Reconnaissance instantanée des chiffres
- 📊 **Visualisation des probabilités** - Barres de confiance pour chaque chiffre
- 🎯 **Haute précision** - Modèle entraîné sur le dataset MNIST complet

## 🚀 Démo Rapide

1. Dessinez un chiffre (0-9) sur le canvas
2. Cliquez sur "Predict" pour obtenir la prédiction
3. Visualisez les probabilités de confiance
4. Utilisez "Clear" pour recommencer

## 🛠️ Technologies Utilisées

### Frontend
- **HTML5 Canvas** - Interface de dessin
- **JavaScript ES6** - Logique applicative
- **Tailwind CSS** - Interface utilisateur moderne
- **ONNX Runtime Web** - Exécution du modèle IA

### Backend/IA
- **PyTorch** - Entraînement du réseau de neurones
- **ONNX** - Format de modèle optimisé
- **NumPy** - Traitement des données
- **Matplotlib** - Visualisation des résultats

## 📦 Installation

### Prérequis
- Node.js (≥16.0.0)
- Python (≥3.8)
- npm ou yarn

### Installation rapide

```bash
# Cloner le repository
git clone <repository-url>
cd project-ia-web-1

# Installer les dépendances JavaScript
npm install

# Installer les dépendances Python
pip install -r requirements.txt

# Lancer l'application
npm run dev
```

L'application sera accessible sur `http://localhost:8000`

## 🎯 Utilisation

### Lancer l'application web
```bash
npm run dev
# ou
npm run serve
```

### Entraîner un nouveau modèle
```bash
npm run train
# ou
cd scripts && python train_mnist.py
```

## 📁 Structure du Projet

```
project-ia-web-1/
├── public/                 # Application web
│   ├── index.html         # Interface principale
│   ├── js/main.js         # Logique JavaScript
│   └── models/            # Modèles IA
│       └── mnist.onnx     # Modèle ONNX pré-entraîné
├── scripts/               # Scripts d'entraînement
│   ├── train_mnist.py     # Entraînement du modèle
│   └── data/              # Données MNIST
├── requirements.txt       # Dépendances Python
├── package.json          # Configuration Node.js
└── README.md             # Documentation
```

## 🔧 Configuration Avancée

### Paramètres d'entraînement
Modifiez `train_mnist.py` pour ajuster :
- Nombre d'époques
- Taille des batches
- Taux d'apprentissage
- Architecture du réseau

### Optimisation Web
- Le modèle ONNX est optimisé pour le navigateur
- Cache automatique du modèle
- Gestion d'erreurs robuste

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Fork le projet
2. Créez votre branche (`git checkout -b feature/amazing-feature`)
3. Committez vos changements (`git commit -m 'Add amazing feature'`)
4. Push vers la branche (`git push origin feature/amazing-feature`)
5. Ouvrez une Pull Request

## 📄 License

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 👨‍💻 Auteur

**Corentin Deleuse**

---

⭐ N'hésitez pas à star le projet si vous le trouvez utile !
